import matplotlib
from matplotlib import font_manager


def load_font_ipaexg():
    ipaexgothic_path = "/Library/Fonts/ipaexg.ttf"
    font_prop = font_manager.FontProperties(fname=ipaexgothic_path)
    matplotlib.rcParams['font.family'] = font_prop.get_name()
    font_manager.fontManager.addfont(ipaexgothic_path)
