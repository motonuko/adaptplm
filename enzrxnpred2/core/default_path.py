import os
import warnings
from functools import cached_property
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def safe_load_env_var_path(env_variable: str) -> Optional[Path]:
    value: Optional[str] = os.getenv(env_variable)
    if value is not None:
        return Path(value)
    else:
        warnings.warn(f"Warning: {env_variable} is not defined. Returning None.")
        return None


class DefaultPath:
    _instance = None

    # singleton
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    @cached_property
    def data_original_dense_screen_processed(self) -> Optional[Path]:
        return safe_load_env_var_path("DATA_ORIGINAL_DENSE_SCREEN_PROCESSED")

    @cached_property
    def data_original_dense_screen_processed_alignments(self) -> Path:
        return self.data_original_dense_screen_processed and self.data_original_dense_screen_processed / 'alignments'

    @cached_property
    def data_original_dense_screen_processed_structure(self) -> Path:
        return self.data_original_dense_screen_processed and self.data_original_dense_screen_processed / 'structure_references'

    @cached_property
    def data_original_kcat_prediction_data(self) -> Path:
        return safe_load_env_var_path("DATA_ORIGINAL_KCAT_PREDICTION_DATA")

    @cached_property
    def data_original_kcat_prediction_kcat_data_splits(self) -> Path:
        return self.data_original_kcat_prediction_data and self.data_original_kcat_prediction_data / 'kcat_data' / 'splits'

    @cached_property
    def data_original_rxnaamapper(self) -> Path:
        return safe_load_env_var_path("DATA_ORIGINAL_RXNAAMAPPER")

    @cached_property
    def data_original_rxnaamapper_predictions(self) -> Path:
        return self.data_original_rxnaamapper and self.data_original_rxnaamapper / 'predictions'

    @cached_property
    def original_esp_fine_tuning_pkl(self):
        return safe_load_env_var_path("ORIGINAL_ESP_FINE_TUNING_PKL")

    @cached_property
    def original_esp_data_dir(self):
        return safe_load_env_var_path("ORIGINAL_ESP_DATA_DIR")

    @cached_property
    def original_kcat_test_pkl(self):
        return safe_load_env_var_path("ORIGINAL_KCAT_TEST_PKL")

    @cached_property
    def original_kcat_train_pkl(self):
        return safe_load_env_var_path("ORIGINAL_KCAT_TRAIN_PKL")

    @cached_property
    def project_root(self) -> Optional[Path]:
        return safe_load_env_var_path("ENZRXNPRED_PROJECT_PATH")

    @cached_property
    def build(self) -> Optional[Path]:
        return self.project_root and self.project_root / 'build'

    @cached_property
    def build_cdhit(self) -> Optional[Path]:
        return self.project_root and self.project_root / 'build' / 'cdhit'

    @cached_property
    def data_dir(self) -> Path:
        return self.project_root and self.project_root / 'data'

    @cached_property
    def data_dataset_dir(self) -> Path:
        return self.project_root and self.project_root / 'data' / 'dataset'

    @cached_property
    def data_dataset_processed(self) -> Path:
        return self.data_dataset_dir and self.data_dataset_dir / 'processed'

    @cached_property
    def data_dataset_raw(self) -> Path:
        return self.data_dataset_dir and self.data_dataset_dir / 'raw'

    @cached_property
    def data_exp_configs_dir(self) -> Path:
        return self.project_root and self.project_root / 'data' / 'exp_configs'

    @cached_property
    def local(self) -> Path:
        return self.project_root and self.project_root / 'local'

    @cached_property
    def test_data_dir(self) -> Path:
        return self.project_root and self.project_root / 'tests' / 'data'

    @cached_property
    def local_exp_rxn_encoder_train(self) -> Path:
        return self.local and self.local / 'exp' / 'rxn_encoder_train'

    @cached_property
    def local_exp_rxn_encoder_eval(self) -> Path:
        return self.local and self.local / 'exp' / 'rxn_encoder_eval'

    @cached_property
    def local_exp_seqrxn_encoder_train(self) -> Path:
        return self.local and self.local / 'exp' / 'seqrxn_encoder_train'

    @cached_property
    def local_exp_seqrxn_encoder_eval(self) -> Path:
        return self.local and self.local / 'exp' / 'seqrxn_encoder_model_eval'

    @cached_property
    def local_exp_nested_cv_train(self) -> Path:
        return self.local and self.local / 'exp' / 'cpi'
