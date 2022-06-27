from typing import Optional, Tuple
import pandas as pd
from nasa_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var not in config.model_config.categorical_vars_with_na + config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    # cast numerical variables as floats
    input_data["est_diameter_min"] = input_data["est_diameter_min"].astype("float")
    input_data["est_diameter_max"] = input_data["est_diameter_max"].astype("float")

    input_data.drop(labels=config.model_config.variables_to_drop, axis=1, inplace=True)

    # Columns should coinside with config.model_config.feature
    assert input_data.columns.tolist() == config.model_config.features

    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(input_data=relevant_data)

    errors = None
    return validated_data, errors
