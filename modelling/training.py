#%%
import pandas as pd
from pathlib import Path

def get_final_training_dataset(datadir_path=Path("G:/My Drive/Data/naver_search_results/")):
    labelled = pd.read_parquet(datadir_path / "navermap_reviews_labelled_only.parquet",
                            engine="pyarrow")
    restaurants = pd.read_parquet(datadir_path / "restaurants_table.parquet",
                                engine="pyarrow",
                                columns=["naver_store_id", "category"])
    with_category = pd.merge(labelled, restaurants,
            left_on="store_id", right_on="naver_store_id", how="left")
    with_category.drop(columns=["naver_store_id"], inplace=True)
    return with_category


