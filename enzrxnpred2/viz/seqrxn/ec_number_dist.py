import math
from collections import Counter
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt

from enzrxnpred2.core.default_path import DefaultPath
from enzrxnpred2.core.package_version import get_package_major_version
from enzrxnpred2.data.enz_seq_rxn_datasource import load_enz_seq_rxn_datasource
from enzrxnpred2.extension.matplotlib_ext import load_font_ipaexg


# deprecated. Use domain.ec_number.to_ec_class
def _to_ec1(ec_number_or_nan: Optional[str]):
    if isinstance(ec_number_or_nan, float) and math.isnan(ec_number_or_nan):
        return 'undefined'
    else:
        return ec_number_or_nan[:1]


def draw_ec_number_distribution(data_path: Path, output_dir: Path, count_reversible_as_one_entry=False,
                                label_in_jp=False):
    load_font_ipaexg()

    df = load_enz_seq_rxn_datasource(data_path)
    if count_reversible_as_one_entry:
        df['data_id_without_direction'] = df['data_id'].apply(lambda x: x[:-3])
        df = df.drop_duplicates(subset=['data_id_without_direction'], keep='first')
    ec1 = df['ec_number'].apply(_to_ec1)
    count = Counter(ec1)
    sorted_keys = sorted([key for key in count.keys() if key != 'undefined']) + ['undefined']
    # labels = [f"EC{k}" if k != 'undefined' else '未定義' for k in sorted_keys]
    sizes = [count[key] for key in sorted_keys]

    if label_in_jp:
        legends = [
            "EC1: 酸化還元酵素",
            "EC2: 転移酵素",
            "EC3: 加水分解酵素",
            "EC4: 脱離酵素",
            "EC5: 異性化酵素",
            "EC6: 合成酵素",
            "EC7: 輸送酵素",
            "未定義"
        ]
    else:
        legends = [
            "EC1: Oxidoreductases",
            "EC2: Transferases",
            "EC3: Hydrolases",
            "EC4: Lyases",
            "EC5: Isomerases",
            "EC6: Ligases",
            "EC7: Translocases",
            "Undefined"
        ]

    def autopct_format(pct, all_values):
        absolute = int(round(pct / 100. * sum(all_values)))
        return f'{absolute} ({pct:.1f}%)'

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(sizes,
                                       # labels=labels,
                                       autopct=lambda pct: autopct_format(pct, sizes),
                                       startangle=90, counterclock=False,
                                       # pctdistance=0.76,
                                       # textprops={'fontsize': 12}
                                       )
    plt.legend(legends, loc='lower right')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    for text in texts:
        text.set_fontsize(13)
        # text.set_fontweight('bold')
    label_text = 'jp' if label_in_jp else 'en'
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"ec_dist_{label_text}_{count_reversible_as_one_entry}.pdf", format="pdf")
    plt.show()


if __name__ == '__main__':
    v = get_package_major_version()
    draw_ec_number_distribution(data_path=DefaultPath().data_dataset_processed / 'enzsrp_full_cleaned.csv',
                                output_dir=DefaultPath().build / 'fig' / v / 'enzsrp_full')
