import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import json
import numpy as np

def create_text_placeholder_chart(
    title: str = "Placeholder Chart",
    text_message: str = "Chart content will go here.",
    height: int = 350
):
    """
    Creates a Plotly figure that displays only text as a placeholder.
    """
    fig = go.Figure()

    # Set the title of the figure
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5, # Center the title horizontally
            xanchor='center',
            font=dict(size=18, color='#333333') # Darker color for readability
        ),
        height=height, # Set the height of the figure
        margin=dict(l=20, r=20, t=60, b=20), # Adjust margins to make space for title and text
        plot_bgcolor='white', # Background color of the plotting area
        paper_bgcolor='#f8f9fa' # Background color of the entire paper/figure (light grey)
    )

    # Add an annotation to display the main text message in the center
    fig.add_annotation(
        xref="paper", yref="paper", # Reference the entire plot area
        x=0.5, y=0.5, # Position at the center of the plot area
        text=text_message,
        showarrow=False, # Do not show an arrow pointing to the text
        font=dict(size=24, color="#cccccc"), # Large, muted text for placeholder feel
        xanchor='center',
        yanchor='middle'
    )

    # Hide all axes, grid lines, and tick labels for a clean placeholder look
    fig.update_xaxes(
        showgrid=False, zeroline=False, showticklabels=False,
        title_text="" # Remove axis title
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, showticklabels=False,
        title_text="" # Remove axis title
    )

    return fig

def visualize_restaurant_nlp_insights(features_df):
    """
    Generates and displays NLP-driven visualizations for a selected restaurant.
    """
    st.subheader("자주 등장하는 명사와 명사를 수식하는 행동과 서술")

    if features_df.empty:
        # 해당 식당에 대해 명사 분석을 한 기록을 찾을 수 없음.
        return None

    restaurant_data = features_df.iloc[0]
    restaurant_name = restaurant_data['restaurant_name']
    selected_restaurant_id = restaurant_data['restaurant_id']
    
    extracted_features_raw = restaurant_data['extracted_features']

    extracted_features = [] # Initialize as empty list for fallback

    # Parsing features
    if isinstance(extracted_features_raw, np.ndarray):
        temp_extracted_features = extracted_features_raw.tolist()

        if temp_extracted_features and isinstance(temp_extracted_features[0], str):
            try:
                extracted_features = json.loads(temp_extracted_features[0])
            except json.JSONDecodeError as e:
                st.warning(f"Could not parse JSON string found in numpy array for {restaurant_name}: {e}. Data might be malformed.")
                extracted_features = []
            except IndexError: # In case temp_extracted_features was empty
                extracted_features = []
        elif temp_extracted_features and isinstance(temp_extracted_features[0], dict):
            extracted_features = temp_extracted_features
        else:
            st.warning(f"Numpy array for {restaurant_name} contained unexpected or empty data after tolist().")
            extracted_features = []

    elif isinstance(extracted_features_raw, str):
        try:
            extracted_features = json.loads(extracted_features_raw)
        except json.JSONDecodeError as e:
            st.warning(f"Could not parse extracted features for {restaurant_name} as JSON: {e}. Data might be malformed.")
            extracted_features = []
    elif extracted_features_raw is None:
        extracted_features = []
    elif isinstance(extracted_features_raw, list):
        extracted_features = extracted_features_raw

    if not extracted_features:
        st.info(f"{restaurant_name}에 대해 유의미한 명사 분석 정보를 추출할 수 없었습니다.")
        return None

    # --- Nouns Bar Chart Data Prep ---
    nouns_data_for_df = []
    for feature in extracted_features:
        nouns_data_for_df.append({
            "noun": feature["noun"],
            "total_count": feature["total_count"]
        })
    
    nouns_df = pd.DataFrame(nouns_data_for_df).sort_values(by="total_count", ascending=False).head(10)

    if nouns_df.empty:
        st.info(f"No dominant nouns found for {restaurant_name}.")
        return

    # Initialize or update session state for selected noun (before chart creation)
    if 'selected_noun_from_chart' not in st.session_state:
        st.session_state.selected_noun_from_chart = None

    # --- Horizontal Layout for Top Section ---
    col1, col2 = st.columns([0.4, 0.6]) # Allocate 40% for nouns, 60% for details

    with col1:
        st.markdown("##### Top 10 자주 사용되었던 명사 (막대를 클릭해보세요!)")
        fig_nouns = px.bar(
            nouns_df,
            y="noun",
            x="total_count",
            orientation='h',
            title=f"{restaurant_name}의 리뷰에서 자주 사용된 명사",
            labels={"noun": "명사", "total_count": "빈도"},
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_nouns.update_layout(yaxis={'categoryorder':'total ascending'})

        chart_event_data = st.plotly_chart(
            fig_nouns,
            key=f"noun_bar_chart_{selected_restaurant_id}",
            on_select="rerun",
            use_container_width=True,
            text_auto=True,
            config={'displayModeBar': False} 
        )

        selected_noun_lemma = None
        if chart_event_data.selection and chart_event_data.selection.points:
            selected_noun_lemma = chart_event_data.selection.points[0].get('y')

        if selected_noun_lemma:
            st.session_state.selected_noun_from_chart = selected_noun_lemma
    
    with col2:
        # --- Noun Details Expander ---
        if st.session_state.selected_noun_from_chart:
            st.markdown("##### 선택된 명사를 수식한 단어들: ")
            st.markdown(f"###### **'{st.session_state.selected_noun_from_chart}'**") # Moved title to be more compact
            
            selected_noun_details = next((item for item in extracted_features if item["noun"] == st.session_state.selected_noun_from_chart), None)

            if selected_noun_details:
                with st.expander(f"'{st.session_state.selected_noun_from_chart}'를 수식한 단어 보기/숨기기", expanded=True): 
                    st.markdown(f"**'{st.session_state.selected_noun_from_chart}' 총 빈도:** {selected_noun_details['total_count']}")

                    st.markdown("###### Top 5 서술:")
                    if selected_noun_details["descriptions"]:
                        desc_df = pd.DataFrame(selected_noun_details["descriptions"]).sort_values(by="count", ascending=False).head(5)
                        fig_desc = px.bar(
                            desc_df, x="count", y="desc", orientation='h',
                            labels={"desc": "서술", "count": "빈도"},
                            color_discrete_sequence=px.colors.qualitative.Dark24
                        )
                        fig_desc.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, 
                                              margin=dict(l=0, r=0, t=30, b=0), height=200)
                        fig_desc.update_traces(marker_color='lightblue')
                        st.plotly_chart(fig_desc, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info(f"'{st.session_state.selected_noun_from_chart}'에 대한 서술 찾을 수 없음.")

                    st.markdown("###### Top 5 행동:")
                    if selected_noun_details["actions"]:
                        action_df = pd.DataFrame(selected_noun_details["actions"]).sort_values(by="count", ascending=False).head(5)
                        fig_action = px.bar(
                            action_df, x="count", y="action", orientation='h',
                            labels={"action": "행동", "count": "빈도"},
                            text_auto=True,
                            color_discrete_sequence=px.colors.qualitative.Dark24
                        )
                        fig_action.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, 
                                               margin=dict(l=0, r=0, t=30, b=0), height=200)
                        fig_action.update_traces(marker_color='lightcoral')
                        st.plotly_chart(fig_action, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info(f"'{st.session_state.selected_noun_from_chart}'에 대한 행동 찾을 수 없음.")
            else:
                st.info("명사를 서술한 단어, 명사가 행하거나 당한 행동에 대해 더 알고 싶으면 해당 명사의 막대를 눌러보세요!")

        else:
            st.info("명사를 서술한 단어, 명사가 행하거나 당한 행동에 대해 더 알고 싶으면 해당 명사의 막대를 눌러보세요!")

    # --- Network Graph of Top Nouns and Related Words (Full Width Below) ---
    st.markdown("---") # Separator between sections
    st.markdown("##### 명사와 수식어 관계 그래프")
    st.info("위의 막대 그래프에서 주목하고 싶은 명사의 막대를 눌러주세요!")

    G = nx.Graph()

    selected_noun_for_graph_highlight = st.session_state.get('selected_noun_from_chart')

    NOUN_SIZE_MULTIPLIER = 2.0
    OTHER_SIZE_MULTIPLIER = 0.7
    MIN_NODE_SIZE = 10
    MAX_NODE_SIZE = 75

    for feature in extracted_features:
        noun = feature["noun"]
        if noun not in nouns_df['noun'].tolist():
            continue

        count = feature["total_count"]
        noun_node_display_size = min(MAX_NODE_SIZE, max(MIN_NODE_SIZE, count * NOUN_SIZE_MULTIPLIER))
        
        noun_color = 'skyblue'
        if selected_noun_for_graph_highlight and noun == selected_noun_for_graph_highlight:
            noun_color = 'gold'

        G.add_node(noun, display_size=noun_node_display_size, color=noun_color, type='noun', count=count)

        sorted_descriptions = sorted(feature["descriptions"], key=lambda x: x["count"], reverse=True)[:5]
        for desc_data in sorted_descriptions:
            desc = desc_data["desc"]
            desc_count = desc_data["count"]
            desc_node_display_size = max(MIN_NODE_SIZE, desc_count * OTHER_SIZE_MULTIPLIER)
            
            desc_color = 'lightgreen'
            if selected_noun_for_graph_highlight and noun == selected_noun_for_graph_highlight:
                desc_color = 'orange'

            G.add_node(desc, display_size=desc_node_display_size, color=desc_color, type='description', count=desc_count)
            
            edge_color = 'gray'
            edge_width = 0.5
            if selected_noun_for_graph_highlight and noun == selected_noun_for_graph_highlight:
                edge_color = 'red'
                edge_width = 1.5

            G.add_edge(noun, desc, weight=desc_count, color=edge_color, width=edge_width)

        sorted_actions = sorted(feature["actions"], key=lambda x: x["count"], reverse=True)[:5]
        for action_data in sorted_actions:
            action = action_data["action"]
            action_count = action_data["count"]
            action_node_display_size = max(MIN_NODE_SIZE, action_count * OTHER_SIZE_MULTIPLIER)
            
            action_color = 'lightcoral'
            if selected_noun_for_graph_highlight and noun == selected_noun_for_graph_highlight:
                action_color = 'purple'

            G.add_node(action, display_size=action_node_display_size, color=action_color, type='action', count=action_count)
            
            edge_color = 'gray'
            edge_width = 0.5
            if selected_noun_for_graph_highlight and noun == selected_noun_for_graph_highlight:
                edge_color = 'red'
                edge_width = 1.5

            G.add_edge(noun, action, weight=action_count, color=edge_color, width=edge_width)

    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

        highlighted_edge_x = []
        highlighted_edge_y = []
        non_highlighted_edge_x = []
        non_highlighted_edge_y = []

        for u, v, edge_attrs in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            if edge_attrs.get('color') == 'red':
                highlighted_edge_x.extend([x0, x1, None])
                highlighted_edge_y.extend([y0, y1, None])
            else:
                non_highlighted_edge_x.extend([x0, x1, None])
                non_highlighted_edge_y.extend([y0, y1, None])

        edge_trace_non_highlight = go.Scatter(
            x=non_highlighted_edge_x, y=non_highlighted_edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Non-highlighted Edges'
        )

        edge_trace_highlight = go.Scatter(
            x=highlighted_edge_x, y=highlighted_edge_y,
            line=dict(width=1.5, color='red'),
            hoverinfo='none',
            mode='lines',
            name='Highlighted Edges'
        )

        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"<b>{node}</b><br>Count: {G.nodes[node]['count']}")
            node_colors.append(G.nodes[node]['color'])
            node_sizes.append(G.nodes[node]['display_size'] * 1.5) # Keep your intended scaling

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=list(G.nodes()),
            textposition="top center",
            textfont=dict(size=10),
            marker=dict(
                showscale=False,
                colorscale='YlGnBu',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                line_width=2
            ),
            name='Nodes'
        )

        fig_graph = go.Figure(data=[edge_trace_non_highlight, edge_trace_highlight, node_trace],
                     layout=go.Layout(
                        title=dict(
                            text=f"'{restaurant_name}'의 리뷰에서 자주 등장한 명사와 수식하는 단어들",
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=50,r=50,t=50),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
        st.plotly_chart(fig_graph, key=f"network_graph_{selected_restaurant_id}", on_select="rerun", use_container_width=True)
    else:
        st.info("Not enough data to create a network graph for this restaurant.")

# # --- Streamlit Application Entry Point ---

# if __name__ == "__main__":
#     st.set_page_config(layout="wide")
#     st.title("Restaurant NLP Insights Dashboard")

#     conn = duckdb.connect(DATABASE_PATH)
#     available_restaurants_df = conn.execute("SELECT naver_store_id, naver_store_name FROM restaurants").fetchdf()
#     conn.close()

#     if not available_restaurants_df.empty:
#         restaurant_options = {row['naver_store_name']: row['naver_store_id'] for index, row in available_restaurants_df.iterrows()}
#         sorted_restaurant_names = sorted(list(restaurant_options.keys()))
        
#         selected_restaurant_name = st.selectbox(
#             "Select a Restaurant:",
#             sorted_restaurant_names,
#             key="main_restaurant_selector"
#         )
#         selected_restaurant_id_from_ui = restaurant_options[selected_restaurant_name]

#         # Reset selected noun in session state when a new restaurant is chosen
#         if st.session_state.get('last_restaurant_id') != selected_restaurant_id_from_ui:
#             st.session_state.selected_noun_from_chart = None
#             st.session_state.last_restaurant_id = selected_restaurant_id_from_ui

#         with st.container(border=True):
#             st.markdown("#### Detailed NLP Analysis for Selected Restaurant")
#             visualize_restaurant_nlp_insights(selected_restaurant_id_from_ui)
            
#     else:
#         st.warning("No restaurants found in the database to visualize. Please run the NLP processing script first.")