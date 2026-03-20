import itertools as it
from pathlib import Path
from typing import Iterable
import re

from dp.phonemizer import Phonemizer
import kokoro as ko
import numpy as np
import piper as pp
import pronouncing as pr
import soundfile as sf
import torch as to
import torchaudio as ta
import tqdm as tq

from .config import Config
from .data_manager import DataManager
from .logger import Logger


class TTS:

    PHONEMES = [
        'AA', 'AE', 'AH', 'AO', 'AW',
        'AX', 'AXR', 'AY', 'EH', 'ER',
        'EY', 'IH', 'IX', 'IY', 'OW',
        'OY', 'UH', 'UW', 'UX'
    ]

    KO_VOICES = [
        'af_heart', 'af_alloy', 'af_aoede', 'af_bella', 'af_jessica',
        'af_kore', 'af_nicole', 'af_nova', 'af_river', 'af_sarah',
        'af_sky', 'am_adam', 'am_echo', 'am_eric', 'am_fenrir',
        'am_liam', 'am_michael', 'am_onyx', 'am_puck', 'am_santa'
    ]
    N_KO_VOICES = 20

    PIPER_MODEL = 'en_US-libritts-high.onnx'
    N_PP_VOICES = 480

    N_VOICES = N_KO_VOICES + N_PP_VOICES

    N_POS_RESAMPLERS = 10
    N_NEG_RESAMPLERS = 1

    N_POS_PER_SPEED = N_VOICES * N_POS_RESAMPLERS
    N_NEG_PER_SPEED = N_VOICES * N_NEG_RESAMPLERS

    N_NEG_TRAIN_SPEEDS = 4
    N_NEG_TEST_SPEEDS = 1
    
    N_NEG_TRAIN_VARS = N_VOICES * N_NEG_RESAMPLERS * N_NEG_TRAIN_SPEEDS
    N_NEG_TEST_VARS = N_VOICES * N_NEG_RESAMPLERS * N_NEG_TEST_SPEEDS


    def __init__ (self, dm: DataManager):
        self.dm = dm
        self.ko_pipeline = ko.KPipeline(
            lang_code = 'a',
            repo_id = 'hexgrad/Kokoro-82M'
        )
        self.pp_pipeline = pp.PiperVoice.load(
            str(dm.MODEL_PATH / TTS.PIPER_MODEL),
            use_cuda = True
        )


    ### PROPERTIES ###
    @property
    def negative_kokoro_resamplers (self) -> list[ta.transforms.Resample]:
        return [ta.transforms.Resample(24000, 16000)]
    
    @property
    def negative_piper_resamplers (self) -> list[ta.transforms.Resample]:
        return [ta.transforms.Resample(22050, 16000)]
    
    @property
    def positive_kokoro_resamplers (self) -> list[ta.transforms.Resample]:
        return [
            ta.transforms.Resample(
                24000, 14000 + i * (4000 // TTS.N_POS_RESAMPLERS)
            ) for i in range(TTS.N_POS_RESAMPLERS)
        ]

    @property
    def positive_piper_resamplers (self) -> list[ta.transforms.Resample]:
        return [
            ta.transforms.Resample(
                22050, 14000 + i * (4000 // TTS.N_POS_RESAMPLERS)
            ) for i in range(TTS.N_POS_RESAMPLERS)
        ]


    ### HELPERS ###
    def _chunks_to_tensor (
                self,
                chunks: Iterable[pp.AudioChunk]
            ) -> to.FloatTensor:
        audio = b''.join([chunk.audio for chunk in chunks])
        audio_int = np.frombuffer(audio, dtype=np.int16)
        wave = to.from_numpy(audio_int).float() / 32768.0
        return wave.unsqueeze(0)


    def _generate_adversarial_phrases (
                self,
                input_phrases: list[str],
                n_adversarial_phrases: int,
                include_partial_phrases: float = 1.0,
                include_input_words: float = 0.2
            ) -> list[str]:
        """
        Generate adversarial words and phrases based on phoneme overlap.
        Currently only works for English phrases. Note that homophones are
        excluded, as this wouldn't actually be an adversarial example for the
        input phrases.

        Args:
            input_phrases (`str`): 
                The target phrases to use for generating for adversarial
                phrases.
            n_adversarial_phrases (`int`):
                The total number of adversarial phrases to return. Uses
                sampling, so not all possible combinations will be included and
                some duplicates may be present.
            include_partial_phrases (`float`):
                The probability of returning a number of words less than the
                input phrase(s) - adversarial phrases always include between
                one and the number of words in a given input phrase.
            include_input_words (`float`):
                The probability of including individual input words in the
                adversarial phrases when the input phrase consists of multiple
                words. For example, if the input phrase was "ok google", then
                setting `include_input_words > 0.0` will allow for adversarial
                phrases like "ok noodle", versus the word "ok" never being
                present in the adversarial texts.

        Returns:
            **adversarial_phrases** (`list[str]`):
                A list of strings corresponding to words and phrases that are
                phonetically similar (but not identical) to the input text.
        """

        # Setup
        all_words = list(set(' '.join(input_phrases).split()))
        word_adversaries: dict[str, list[str]] = {}
        phonemizer = Phonemizer.from_checkpoint(
            str(DataManager.MODEL_PATH / 'en_us_cmudict_forward.pt')
        )

        # Get adversaries for each word
        for word in all_words:
            # Get base phones
            phones = pr.phones_for_word(word)
            if not phones:
                raw_phs = phonemizer(word, lang='en_us')
                phones = re.sub(f'[\]|[]', '', re.sub(f'\]\[', ' ', raw_phs))
            else:
                phones: str = phones[0]
                
            # Add lexical stress patterns to vowels
            phones_with_stress = [
                re.sub(
                    '|'.join(TTS.PHONEMES),
                    lambda x: f"{x.group(0)}[0|1|2]",
                    re.sub(r'\d+', '', p)
                ) for p in phones
            ]

            # Build search queries
            query_exps: list[str] = []
            current = phones_with_stress[0].split()
            if len(current) <= 2:
                query_exps.append(' '.join(current))
            else:
                query_exps.extend(
                    self._phoneme_replacement(
                        current,
                        max(0, len(current) - 2)
                    )
                )

            # Find phonetic matches
            adversaries: list[str] = []
            for query in query_exps:
                matches: list[str] = pr.search(query)
                for m in matches:
                    m_phones = pr.phones_for_word(m)
                    if m_phones and m_phones[0] != current and \
                            m.lower() != word.lower():
                        adversaries.append(m)

            word_adversaries[word] = list(set(adversaries))

        # Get unique adversarial phrases
        unique_adversarial_pool: set[str] = set()
        for _ in range(n_adversarial_phrases * 2):
            target: str = np.random.choice(input_phrases)
            target_words = target.split()

            mutated: list[str] = []
            for w in target_words:
                if np.random.random() < include_input_words:
                    mutated.append(w)
                else:
                    candidates = word_adversaries.get(w, [])
                    mutated.append(
                        np.random.choice(candidates) if candidates else w
                    )

            if include_partial_phrases > 0 and len(mutated) > 1:
                if np.random.random() <= include_partial_phrases:
                    n_subset = np.random.randint(1, len(mutated) + 1)
                    mutated = list(
                        np.random.choice(mutated, n_subset, replace=False)
                    )

            out_phrase = ' '.join(mutated)
            if out_phrase not in input_phrases:
                unique_adversarial_pool.add(out_phrase)

        # Sample unique adversarials n_adversarial_phrases times
        unique_adversarials: list[str] = list(unique_adversarial_pool)
        if len(unique_adversarials) >= n_adversarial_phrases:
            return list(np.random.choice(
                unique_adversarials,
                n_adversarial_phrases,
                replace=False
            ))
        else:
            n_dups = n_adversarial_phrases - len(unique_adversarials)
            dups = list(
                np.random.choice(unique_adversarials, n_dups, replace=True)
            )
            return unique_adversarials + dups

    
    def _generate_batch (
                self,
                output: Path,
                phrases: list[str],
                speeds: list[float],
                ko_resamplers: list[ta.transforms.Resample],
                pp_resamplers: list[ta.transforms.Resample],
                name: str
            ):
        Logger.log(f'🔄 generating {name} samples...')
        n = TTS.N_VOICES * len(phrases) * len(speeds) * len(ko_resamplers)
        pbar = tq.tqdm(total=n, desc=f'Generating {name} samples')
        idx = 0
        for p, v, s in it.product(phrases, TTS.KO_VOICES, speeds):
            generator = self.ko_pipeline(p, v, s)
            _, _, audio = next(generator)
            for resampler in self.resamplers:
                resampled = resampler(audio)
                path = str(output / f'sample_{idx}.wav')
                sf.write(path, resampled, 16000)
                pbar.update()
                idx += 1
        for v, s in it.product(range(TTS.N_PP_VOICES), speeds):
            config = pp.SynthesisConfig(v, s)
            for p, resampler in it.product(phrases, pp_resamplers):
                chunks = self.pp_pipeline.synthesize(p, config)
                audio = self._chunks_to_tensor(chunks)
                resampled = resampler(audio)
                path = str(output / f'sample_{idx}.wav')
                sf.write(path, resampled, 16000)
                pbar.update()
                idx += 1


    def _phoneme_replacement(
                self,
                input_chars: str,
                max_replace: int,
                replace_char = '"(.){1,3}"'
            ) -> list[str]:
        results: list[str] = []
        chars = list(input_chars)
        for r in range(1, max_replace+1):
            comb = it.combinations(range(len(chars)), r)
            for indices in comb:
                chars_copy = chars.copy()
                for i in indices:
                    chars_copy[i] = replace_char
                results.append(' '.join(chars_copy))
        return results
    

    def _speeds (self, n: int) -> list[float]:
        return list(np.linspace(0.7, 1.3, n))


    ### METHODS ###
    def generate (self, config: Config):
        Logger.log('🚀 starting sample generation...')

        paths = [
            self.dm.pos_train,
            self.dm.pos_test,
            self.dm.neg_train,
            self.dm.neg_test
        ]

        n_adv_train = (config.n_train // TTS.N_NEG_TRAIN_VARS)
        adv_train = self._generate_adversarial_phrases(
            config.target_phrases,
            n_adv_train - len(config.negative_phrases)
        )

        n_adv_test = (config.n_test // TTS.N_NEG_TEST_VARS)
        adv_test = list(np.random.choice(
            adv_train,
            n_adv_test - len(config.negative_phrases),
            replace=False
        ))

        all_phrases = [
            config.target_phrases,
            config.target_phrases,
            config.negative_phrases + adv_train,
            config.negative_phrases + adv_test
        ]

        n_pos_train_per_phrase = config.n_train // len(config.target_phrases)
        n_pos_test_per_phrase = config.n_test // len(config.target_phrases)

        pos_train_speeds = self._speeds(
            n_pos_train_per_phrase // TTS.N_POS_PER_SPEED
        )
        pos_test_speeds = self._speeds(
            n_pos_test_per_phrase // TTS.N_POS_PER_SPEED
        )

        all_speeds = [
            pos_train_speeds,
            pos_test_speeds,
            self._speeds(TTS.N_NEG_TRAIN_SPEEDS),
            self._speeds(TTS.N_NEG_TEST_SPEEDS)
        ]

        all_ko_resamplers = [
            self.positive_kokoro_resamplers,
            self.positive_kokoro_resamplers,
            self.negative_kokoro_resamplers,
            self.negative_kokoro_resamplers
        ]

        all_pp_resamplers = [
            self.positive_piper_resamplers,
            self.positive_piper_resamplers,
            self.negative_piper_resamplers,
            self.negative_piper_resamplers
        ]

        names = [
            'positive training',
            'positive testing',
            'negative training',
            'negative testing'
        ]

        zipped = zip(
            paths,
            all_phrases,
            all_speeds,
            all_ko_resamplers,
            all_pp_resamplers,
            names
        )

        for tup in zipped:
            self._generate_batch(*tup)

        Logger.log('✨ all samples generated')