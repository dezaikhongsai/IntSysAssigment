from .bmi.bmi import (
    calc_bmi,
    classify_bmi,
    calc_bmi_array,
    classify_bmi_array,
    bmi_table,
    plot_bmi_bar,   # nếu bạn thêm hàm vẽ
)

__all__ = [
    "calc_bmi", "classify_bmi",
    "calc_bmi_array", "classify_bmi_array",
    "bmi_table", "plot_bmi_bar",
]
