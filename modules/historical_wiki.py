# historical_wiki.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import streamlit as st

class F1HistoricalWiki:
    def __init__(self, api):
        self.api = api
        self.timeline_events = {
            "1950": "First F1 World Championship",
            "1958": "First Constructors' Championship",
            "1968": "Introduction of wings",
            "1977": "Ground effect era begins",
            "1994": "Safety reforms after Senna's accident",
            "2000": "Ferrari dominance begins",
            "2009": "Introduction of KERS",
            "2014": "Hybrid era begins",
            "2022": "New regulations & ground effect return"
        }
        
    def create_historical_section(self):
        """Create the historical wiki section of the dashboard"""
        st.header("📚 F1 Historical Encyclopedia", divider="red")
        
        # Create tabs within Historical Wiki
        wiki_tabs = st.tabs(["Timeline", "Champions", "Iconic Circuits", "Technical Evolution"])
        
        with wiki_tabs[0]:
            self.create_timeline_section()
            
        with wiki_tabs[1]:
            self.create_champions_section()
            
        with wiki_tabs[2]:
            self.create_circuits_section()
            
        with wiki_tabs[3]:
            self.create_technical_section()
    

    def create_timeline_section(self):
            """Create improved interactive F1 timeline"""
            st.subheader("🏁 Formula 1 Timeline")
            
            # Create timeline visualization with improved readability
            fig = go.Figure()
            
            years = list(self.timeline_events.keys())
            events = list(self.timeline_events.values())
            
            # Create the timeline with improved text positioning
            fig.add_trace(go.Scatter(
                x=years,
                y=[0] * len(years),
                mode='markers+text',
                marker=dict(
                    size=20,
                    color='#E10600',
                    symbol='diamond'
                ),
                text=events,
                textposition='top center',
                textfont=dict(size=12),
                hoverinfo='text',
                hovertext=[f"{year}: {event}" for year, event in self.timeline_events.items()]
            ))
            
            # Improved layout with angled text and more space
            fig.update_layout(
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    zeroline=False,
                    range=[-1, 2]  # Increase vertical space for labels
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(255,255,255,0.1)',
                    title="Year",
                    tickangle=45  # Angle the year labels
                ),
                font=dict(color='white'),
                height=500,  # Increase height for better readability
                margin=dict(l=20, r=20, t=100, b=100)  # Increase top and bottom margins
            )
            
            # Add alternating text positions to avoid overlap
            for i in range(len(years)):
                fig.add_annotation(
                    x=years[i],
                    y=0.1 if i % 2 == 0 else 0.5,  # Alternate between two heights
                    text=events[i],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#E10600",
                    font=dict(size=12, color="white"),
                    align="left",
                    textangle=0,
                    ax=0,
                    ay=-40 if i % 2 == 0 else -80
                )
            
            st.plotly_chart(fig, use_container_width=True)

    def create_champions_section(self):
        """Display F1 champions information"""
        st.subheader("👑 Hall of Champions")
        
        # Create champions data
        champions_data = {
            "Driver": ["Lewis Hamilton", "Michael Schumacher", "Juan Manuel Fangio",
                      "Alain Prost", "Sebastian Vettel", "Max Verstappen"],
            "Championships": [7, 7, 5, 4, 4, 3],
            "Era": ["2008-2020", "1994-2004", "1951-1957", "1985-1993", "2010-2013", "2021-2023"],
            "Teams": ["McLaren, Mercedes", "Benetton, Ferrari", "Alfa Romeo, Mercedes, Ferrari, Maserati",
                     "McLaren, Ferrari, Williams", "Red Bull", "Red Bull"]
        }
        champions_df = pd.DataFrame(champions_data)
        
        # Create visual championship comparison
        fig = px.bar(champions_df,
                    x="Driver",
                    y="Championships",
                    color="Championships",
                    color_continuous_scale=["#FFD700", "#E10600"],
                    text="Championships")
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display champions details in an expandable table
        with st.expander("📊 View Detailed Champions Data"):
            st.dataframe(
                champions_df,
                column_config={
                    "Driver": st.column_config.TextColumn("Legend", width="medium"),
                    "Championships": st.column_config.NumberColumn("Titles", format="%d 🏆"),
                    "Era": st.column_config.TextColumn("Golden Years", width="small"),
                    "Teams": st.column_config.TextColumn("Constructor(s)", width="large")
                },
                hide_index=True
            )
    
    def create_circuits_section(self):
        """Display iconic F1 circuits information"""
        st.subheader("🏎️ Legendary Circuits")
        
        # Circuit selection
        circuits = {
            "Monaco": {
                "length": 3.337,
                "turns": 19,
                "first_gp": 1950,
                "description": "The ultimate test of precision and concentration, winding through the streets of Monte Carlo.",
                "key_features": ["Casino Square", "Swimming Pool", "Tunnel"]
            },
            "Spa-Francorchamps": {
                "length": 7.004,
                "turns": 20,
                "first_gp": 1950,
                "description": "Known for its unpredictable weather and the infamous Eau Rouge corner.",
                "key_features": ["Eau Rouge", "Raidillon", "Les Combes"]
            },
            "Monza": {
                "length": 5.793,
                "turns": 11,
                "first_gp": 1950,
                "description": "The Temple of Speed, featuring long straights and rich history.",
                "key_features": ["Parabolica", "Lesmo", "Variante Ascari"]
            }
        }
        
        selected_circuit = st.selectbox("Select Circuit", list(circuits.keys()))
        
        # Display circuit information in a modern card layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
                <div style='background-color: rgba(225,6,0,0.1); padding: 20px; border-radius: 10px; border: 1px solid #E10600'>
                    <h3>{selected_circuit}</h3>
                    <p>{circuits[selected_circuit]['description']}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
                <div style='background-color: #2E2E2E; padding: 20px; border-radius: 10px;'>
                    <h4>Circuit Stats</h4>
                    <p>Length: {circuits[selected_circuit]['length']} km</p>
                    <p>Turns: {circuits[selected_circuit]['turns']}</p>
                    <p>First GP: {circuits[selected_circuit]['first_gp']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Display key features
        st.subheader("Key Features")
        for feature in circuits[selected_circuit]['key_features']:
            st.markdown(f"🔹 {feature}")
    
    def create_technical_section(self):
        """Display F1 technical evolution information"""
        st.subheader("⚙️ Technical Evolution")
        
        # Create eras timeline
        eras = {
            "1950s": "Front-engine cars, drum brakes, minimal aerodynamics",
            "1960s": "Rear-engine revolution, introduction of wings",
            "1970s": "Ground effect era begins, increased focus on aerodynamics",
            "1980s": "Turbo era, advanced composites, electronic aids",
            "1990s": "Active suspension, traction control, V10 engines",
            "2000s": "Grooved tires, launch control, V8 engines",
            "2010s": "KERS, DRS, hybrid power units",
            "2020s": "Ground effect return, 18-inch wheels, sustainable fuels"
        }
        
        # Create interactive era explorer
        selected_era = st.select_slider(
            "Explore technical evolution by decade:",
            options=list(eras.keys()),
            value="2020s"
        )
        
        # Display era information in a modern card
        st.markdown(f"""
            <div style='background-color: rgba(225,6,0,0.1); padding: 20px; border-radius: 10px; border: 1px solid #E10600'>
                <h3>{selected_era}</h3>
                <p>{eras[selected_era]}</p>
            </div>
        """, unsafe_allow_html=True)