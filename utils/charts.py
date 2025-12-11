import plotly.express as px
import plotly.graph_objects as go

def create_pareto_chart(df, x_col, y_col):
    fig = px.bar(df, x=x_col, y=y_col)
    return fig

def create_distribution_chart(df, col):
    fig = px.histogram(df, x=col)
    return fig

def create_scatter_personas(df, x_col, y_col, color_col='persona_id'):
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
    return fig

def radar_persona(df, persona_id, features):
    persona_data = df[df['persona_id'] == persona_id]
    fig = go.Figure()
    for feature in features:
        fig.add_trace(go.Scatterpolar(
            r=persona_data[feature].values,
            theta=features,
            fill='toself',
            name=f'Persona {persona_id}'
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True
    )
    return fig
