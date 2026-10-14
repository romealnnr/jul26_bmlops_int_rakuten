"""Names of the 27 Rakuten product categories (prdtypecode -> readable label).

Copied from streamlit_app/config.py so the API can return a category name with
every prediction. The labels are short English descriptions of each code.
"""

from __future__ import annotations

PRDTYPECODE_LABELS: dict[int, str] = {
    10: "Used books",
    40: "Video games / imported CDs",
    50: "Console accessories",
    60: "Game consoles",
    1140: "Figurines / merchandise",
    1160: "Collectible cards",
    1180: "Figurines / role-playing games",
    1280: "Toys / plush toys",
    1281: "Children's board games",
    1300: "Scale models / drones",
    1301: "Baby clothing / children's accessories",
    1302: "Outdoor games / leisure",
    1320: "Baby and childcare products",
    1560: "Furniture / home interior",
    1920: "Household linen",
    1940: "Food",
    2060: "Decoration",
    2220: "Pet supplies",
    2280: "Used magazines",
    2403: "Books / comics / sheet music",
    2462: "Used video games",
    2522: "Stationery",
    2582: "Garden furniture",
    2583: "Pool / spa",
    2585: "Tools / gardening",
    2705: "New books",
    2905: "PC games",
}


def category_name(code: int) -> str:
    """Readable name of a prdtypecode; unknown codes are labelled as such."""
    return PRDTYPECODE_LABELS.get(int(code), f"Unknown category {int(code)}")
