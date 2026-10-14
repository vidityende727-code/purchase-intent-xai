import logging

logger = logging.getLogger(__name__)


def create_features(df):
    df = df.copy()

    df["TotalPagesViewed"] = (
        df["Administrative"] + df["Informational"] + df["ProductRelated"]
    )

    df["TotalDuration"] = (
        df["Administrative_Duration"]
        + df["Informational_Duration"]
        + df["ProductRelated_Duration"]
    )

    df["AvgTimePerPage"] = df["TotalDuration"] / df["TotalPagesViewed"].replace(0, 1)

    logger.info("Feature engineering complete: added 3 new features.")

    return df
