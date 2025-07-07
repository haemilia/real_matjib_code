import duckdb
import spacy
import pandas as pd
from collections import Counter, defaultdict
import os
import json
from tqdm.auto import tqdm # Import tqdm for progress bars

# --- Your existing function definitions (no changes needed here) ---
def extract_noun_dependencies_from_text(review_text, nlp_model):
    """
    Extracts nouns, their descriptions, and associated actions from a single review text
    using a spaCy NLP model.
    """
    doc = nlp_model(review_text)

    noun_counts = Counter()
    noun_desc_counts = defaultdict(Counter)
    noun_action_counts = defaultdict(Counter)

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            noun_lemma = token.lemma_
            noun_counts[noun_lemma] += 1

            for child in token.children:
                if child.dep_ in ["amod", "nmod"]:
                    if child.pos_ in ["ADJ", "VERB", "NOUN"]:
                        noun_desc_counts[noun_lemma][child.lemma_] += 1
            
            # The original code had a slight issue here, `add` is for sets.
            # For Counter, you should use += 1 or update().
            # Let's re-evaluate the action extraction logic to be consistent with Counter.
            # It seems the intention was to increment counts.
            if token.dep_ in ["nsubj", "obj", "iobj"] and token.head.pos_ == "VERB":
                noun_action_counts[noun_lemma][token.head.lemma_] += 1
            elif token.head and token.head.pos_ == "VERB" and token.dep_ == "acl":
                noun_action_counts[noun_lemma][token.head.lemma_] += 1 # Changed from .add()
            elif token.head and token.head.pos_ == "VERB" and token.dep_ not in ["amod", "nmod"]:
                noun_action_counts[noun_lemma][token.head.lemma_] += 1

    return noun_counts, noun_desc_counts, noun_action_counts

def aggregate_restaurant_features_for_duckdb(nlp, input_df):
    """
    Processes reviews to extract linguistic features and then aggregates them
    per restaurant, producing both flat aggregated DataFrames and a nested summary DataFrame
    suitable for DuckDB's complex types.

    Args:
        input_df (pd.DataFrame): A DataFrame with at least 'review_id', 'store_naver_name',
                                 'store_id' (or 'naver_store_id' as the linking ID),
                                 and 'review_text' columns.
                                 Crucially, it expects a consistent 'store_id' or 'naver_store_id'
                                 column that uniquely identifies the restaurant.

    Returns:
        tuple: A tuple of Pandas DataFrames:
               (df_aggregated_nouns, df_aggregated_noun_descriptions,
                df_aggregated_noun_actions, df_restaurant_nested_summary)
               Returns (None, ...) if the spaCy model cannot be loaded.
    """

    all_extracted_review_data = []

    restaurant_id_col = None
    if 'store_id' in input_df.columns:
        restaurant_id_col = 'store_id'
    elif 'naver_store_id' in input_df.columns:
        restaurant_id_col = 'naver_store_id'
    # else: No warning here, as it's handled by current_restaurant_id check

    for index, row in input_df.iterrows():
        restaurant_name = row['store_naver_name']
        review_text = str(row['review_text']).strip()
        
        current_restaurant_id = None
        if restaurant_id_col:
            current_restaurant_id = row[restaurant_id_col]

        if not review_text:
            continue

        noun_counts, noun_desc_counts, noun_action_counts = \
            extract_noun_dependencies_from_text(review_text, nlp)
        
        extracted_data_for_review = {
            'restaurant_name': restaurant_name,
            'noun_counts': noun_counts,
            'noun_desc_counts': noun_desc_counts,
            'noun_action_counts': noun_action_counts
        }
        if current_restaurant_id:
            extracted_data_for_review['restaurant_id'] = current_restaurant_id
        
        all_extracted_review_data.append(extracted_data_for_review)

    temp_df = pd.DataFrame(all_extracted_review_data)

    aggregated_nouns_data = []
    aggregated_noun_descriptions_data = []
    aggregated_noun_actions_data = []
    restaurant_nested_summary_data = []

    group_cols = ['restaurant_name']
    if restaurant_id_col:
        group_cols.insert(0, 'restaurant_id')

    for group_key, group_df in temp_df.groupby(group_cols):
        if restaurant_id_col:
            restaurant_id, restaurant_name = group_key
        else:
            restaurant_name = group_key
            restaurant_id = None

        total_reviews_processed = len(group_df)
        
        combined_noun_counts = Counter()
        combined_noun_desc_counts = defaultdict(Counter)
        combined_noun_action_counts = defaultdict(Counter)

        for _, row in group_df.iterrows():
            combined_noun_counts.update(row['noun_counts'])
            for noun, desc_counter in row['noun_desc_counts'].items():
                combined_noun_desc_counts[noun].update(desc_counter)
            for noun, action_counter in row['noun_action_counts'].items():
                combined_noun_action_counts[noun].update(action_counter)
        
        # --- Populate Flat Aggregated Data ---
        for noun_lemma, freq in combined_noun_counts.items():
            entry = {
                "restaurant_name": restaurant_name,
                "noun_lemma": noun_lemma,
                "total_frequency": freq
            }
            if restaurant_id:
                entry['restaurant_id'] = restaurant_id
            aggregated_nouns_data.append(entry)

            for desc_lemma, desc_freq in combined_noun_desc_counts[noun_lemma].items():
                entry = {
                    "restaurant_name": restaurant_name,
                    "noun_lemma": noun_lemma,
                    "description_lemma": desc_lemma,
                    "total_frequency_in_context": desc_freq
                }
                if restaurant_id:
                    entry['restaurant_id'] = restaurant_id
                aggregated_noun_descriptions_data.append(entry)
            
            for action_lemma, action_freq in combined_noun_action_counts[noun_lemma].items():
                entry = {
                    "restaurant_name": restaurant_name,
                    "noun_lemma": noun_lemma,
                    "action_lemma": action_lemma,
                    "total_frequency_in_context": action_freq
                }
                if restaurant_id:
                    entry['restaurant_id'] = restaurant_id
                aggregated_noun_actions_data.append(entry)
        
        # --- Populate Nested Summary Data ---
        restaurant_nouns_list = []
        sorted_nouns = sorted(combined_noun_counts.items(), key=lambda item: item[1], reverse=True)

        for noun_lemma, noun_total_count in sorted_nouns:
            descriptions_list = []
            if noun_lemma in combined_noun_desc_counts:
                sorted_descriptions = sorted(combined_noun_desc_counts[noun_lemma].items(), 
                                             key=lambda item: item[1], reverse=True)
                for desc_lemma, desc_count in sorted_descriptions:
                    descriptions_list.append({"desc": desc_lemma, "count": desc_count})

            actions_list = []
            if noun_lemma in combined_noun_action_counts:
                sorted_actions = sorted(combined_noun_action_counts[noun_lemma].items(),
                                         key=lambda item: item[1], reverse=True)
                for action_lemma, action_count in sorted_actions:
                    actions_list.append({"action": action_lemma, "count": action_count})
            
            restaurant_nouns_list.append({
                "noun": noun_lemma,
                "total_count": noun_total_count,
                "descriptions": descriptions_list,
                "actions": actions_list
            })
        
        summary_entry = {
            "restaurant_name": restaurant_name,
            "total_reviews_processed": total_reviews_processed,
            "extracted_features": restaurant_nouns_list
        }
        if restaurant_id:
            summary_entry['restaurant_id'] = restaurant_id
        restaurant_nested_summary_data.append(summary_entry)

    df_aggregated_nouns = pd.DataFrame(aggregated_nouns_data)
    df_aggregated_noun_descriptions = pd.DataFrame(aggregated_noun_descriptions_data)
    df_aggregated_noun_actions = pd.DataFrame(aggregated_noun_actions_data)
    df_restaurant_nested_summary = pd.DataFrame(restaurant_nested_summary_data)

    return df_aggregated_nouns, df_aggregated_noun_descriptions, df_aggregated_noun_actions, df_restaurant_nested_summary

def get_restaurants():
    with duckdb.connect(r"H:\My Drive\reviews.db") as con:
        df = con.table("restaurants").fetchdf()
    return df

def get_reviews_per_restaurants(store_id:str):
    with duckdb.connect(r"H:\My Drive\reviews.db") as con:
        query = """
            SELECT
                n.store_id,
                n.review_id,
                n.store_naver_name,
                n.review_text
            FROM
                navermap_reviews AS n
            WHERE
                n.store_id = ? 
        """
        df = con.execute(query, [store_id]).fetchdf()
    return df

# --- Main Script for Processing All Data and Storing to DuckDB ---
if __name__ == "__main__":
    print("--- Starting full data aggregation and storage (NLP processing) ---")
    try:
        nlp = spacy.load("ko_core_news_md")
    except OSError as e:
        print("Korean spaCy model 'ko_core_news_md' not found. Please run: python -m spacy download ko_core_news_md")
        raise e
    # Lists to collect DataFrames from each restaurant
    all_agg_nouns_dfs = []
    all_agg_desc_dfs = []
    all_agg_actions_dfs = []
    all_nested_summary_dfs = []

    # 1. Get the list of all restaurants
    all_restaurants_df = get_restaurants()
    
    print(f"Total restaurants to process: {len(all_restaurants_df)}")

    # 2. Iterate through each restaurant with tqdm progress bar
    pbar = tqdm(all_restaurants_df.iterrows(), 
                total=len(all_restaurants_df), 
                desc="Processing Restaurants")
    for index, restaurant_row in pbar:
        current_restaurant_id = restaurant_row["naver_store_id"]
        current_restaurant_name = restaurant_row["naver_store_name"]

        # Reduce verbosity inside the loop, tqdm will show overall progress
        # print(f"Processing reviews for restaurant: {current_restaurant_name} (ID: {current_restaurant_id})")

        reviews_df = get_reviews_per_restaurants(current_restaurant_id)

        if reviews_df.empty:
            pbar.set_postfix_str(f"Skipped {current_restaurant_name} (No reviews)", refresh=False)
            continue # Silently skip if no reviews

        pbar.set_postfix_str(f"{current_restaurant_name} ({len(reviews_df)} reviews)", refresh=True)

        df_agg_nouns, df_agg_desc, df_agg_actions, df_nested_summary = \
            aggregate_restaurant_features_for_duckdb(nlp, reviews_df)

        if not df_agg_nouns.empty or not df_agg_desc.empty or not df_agg_actions.empty or not df_nested_summary.empty:
            all_agg_nouns_dfs.append(df_agg_nouns)
            all_agg_desc_dfs.append(df_agg_desc)
            all_agg_actions_dfs.append(df_agg_actions)
            all_nested_summary_dfs.append(df_nested_summary)
        else:
            pbar.write(f"Note: No extractable features found for {current_restaurant_name} (ID: {current_restaurant_id}). Skipping aggregation.")

    print("\n--- All individual restaurant aggregations complete. Concatenating results ---")

    # 3. Concatenate all collected DataFrames outside the loop
    final_agg_nouns_df = pd.concat(all_agg_nouns_dfs, ignore_index=True) if all_agg_nouns_dfs else pd.DataFrame()
    final_agg_desc_df = pd.concat(all_agg_desc_dfs, ignore_index=True) if all_agg_desc_dfs else pd.DataFrame()
    final_agg_actions_df = pd.concat(all_agg_actions_dfs, ignore_index=True) if all_agg_actions_dfs else pd.DataFrame()
    final_nested_summary_df = pd.concat(all_nested_summary_dfs, ignore_index=True) if all_nested_summary_dfs else pd.DataFrame()

    print(f"Final Aggregated Nouns DF shape: {final_agg_nouns_df.shape}")
    print(f"Final Aggregated Descriptions DF shape: {final_agg_desc_df.shape}")
    print(f"Final Aggregated Actions DF shape: {final_agg_actions_df.shape}")
    print(f"Final Nested Summary DF shape: {final_nested_summary_df.shape}")

    # 4. Store these final DataFrames into DuckDB # local db for now
    con = duckdb.connect(r"C:\Users\lhi30\Haein\2025\Projects\combined\real_matjib_code\reviews_local.db")

    try:
        # Store restaurant_nouns_aggregated
        if not final_agg_nouns_df.empty:
            print("\nStoring 'restaurant_nouns_aggregated' to DuckDB...")
            con.execute("DROP TABLE IF EXISTS restaurant_nouns_aggregated")
            con.execute("CREATE TABLE restaurant_nouns_aggregated AS SELECT * FROM final_agg_nouns_df")
            print("  Table 'restaurant_nouns_aggregated' created/overwritten.")

        # Store restaurant_descriptions_aggregated
        if not final_agg_desc_df.empty:
            print("Storing 'restaurant_descriptions_aggregated' to DuckDB...")
            con.execute("DROP TABLE IF EXISTS restaurant_descriptions_aggregated")
            con.execute("CREATE TABLE restaurant_descriptions_aggregated AS SELECT * FROM final_agg_desc_df")
            print("  Table 'restaurant_descriptions_aggregated' created/overwritten.")

        # Store restaurant_actions_aggregated
        if not final_agg_actions_df.empty:
            print("Storing 'restaurant_actions_aggregated' to DuckDB...")
            con.execute("DROP TABLE IF EXISTS restaurant_actions_aggregated")
            con.execute("CREATE TABLE restaurant_actions_aggregated AS SELECT * FROM final_agg_actions_df")
            print("  Table 'restaurant_actions_aggregated' created/overwritten.")

        # Store restaurant_feature_summaries (for nested data)
        if not final_nested_summary_df.empty:
            print("Storing 'restaurant_feature_summaries' (nested) to DuckDB...")
            con.execute("DROP TABLE IF EXISTS restaurant_feature_summaries")
            con.execute("CREATE TABLE restaurant_feature_summaries AS SELECT * FROM final_nested_summary_df")
            print("  Table 'restaurant_feature_summaries' created/overwritten.")
            
            print("\nSchema of restaurant_feature_summaries:")
            print(con.execute("PRAGMA table_info('restaurant_feature_summaries')").fetchdf())

    except Exception as e:
        print(f"An error occurred while storing data to DuckDB: {e}")
    finally:
        con.close()
        print("\nDuckDB connection closed for storage process.")

    print("\n--- Full data aggregation and storage complete ---")