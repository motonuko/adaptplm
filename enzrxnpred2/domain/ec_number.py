import math
from typing import Optional


def to_ec_class(ec_number_or_nan: Optional[str]):
    if isinstance(ec_number_or_nan, float) and math.isnan(ec_number_or_nan):
        return ec_number_or_nan
    else:
        return ec_number_or_nan[:1]
