# driver_insights.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import streamlit as st

class F1DriverInsights:
    def __init__(self, api):
        self.api = api
        
    def create_driver_insights_section(self):
        """Create the driver insights section of the dashboard"""
        st.header("👨‍🚀 Driver Insights Hub", divider="red")
        
        # Create tabs within Driver Insights
        insight_tabs = st.tabs(["Performance Analysis", "Head-to-Head", "Career Stats", "Style Analysis"])
        
        with insight_tabs[0]:
            self.create_performance_analysis()
            
        with insight_tabs[1]:
            self.create_head_to_head_comparison()
            
        with insight_tabs[2]:
            self.create_career_stats()  
            
        with insight_tabs[3]:
            self.create_driving_style_analysis()


    def create_career_stats(self):
        """Create career statistics section"""
        st.subheader("📈 Career Statistics")
        
        # Driver selection
        selected_driver = st.selectbox(
            "Select Driver for Career Analysis",
            ["Lewis Hamilton", "Max Verstappen", "Fernando Alonso"]
        )
        
        # Career stats data structure
        career_stats = {
            "Lewis Hamilton": {
                "Seasons": 17,
                "Championships": 7,
                "Race Entries": 332,
                "Race Wins": 103,
                "Podiums": 197,
                "Poles": 104,
                "Fastest Laps": 64,
                "Points": 4639.5,
                "First Win": "2007 Canadian Grand Prix",
                "Last Win": "2021 Saudi Arabian Grand Prix",
                "Teams": ["McLaren", "Mercedes"]
            },
            "Max Verstappen": {
                "Seasons": 9,
                "Championships": 3,
                "Race Entries": 185,
                "Race Wins": 54,
                "Podiums": 98,
                "Poles": 32,
                "Fastest Laps": 27,
                "Points": 2586.5,
                "First Win": "2016 Spanish Grand Prix",
                "Last Win": "2024 Saudi Arabian Grand Prix",
                "Teams": ["Toro Rosso", "Red Bull"]
            },
            "Fernando Alonso": {
                "Seasons": 20,
                "Championships": 2,
                "Race Entries": 377,
                "Race Wins": 32,
                "Podiums": 106,
                "Poles": 22,
                "Fastest Laps": 23,
                "Points": 2267,
                "First Win": "2003 Hungarian Grand Prix",
                "Last Win": "2013 Spanish Grand Prix",
                "Teams": ["Minardi", "Renault", "McLaren", "Ferrari", "Alpine", "Aston Martin"]
            }
        }
        
        stats = career_stats[selected_driver]
        
        # Create modern stat cards layout
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div style='background: linear-gradient(145deg, #2E2E2E, #1A1A1A); padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #E10600;'>Championships 🏆</h3>
                    <h2 style='color: white; margin: 0;'>{}</h2>
                </div>
            """.format(stats["Championships"]), unsafe_allow_html=True)
            
        with col2:
            win_rate = round(stats["Race Wins"] / stats["Race Entries"] * 100, 1)
            st.markdown("""
                <div style='background: linear-gradient(145deg, #2E2E2E, #1A1A1A); padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #E10600;'>Win Rate 📊</h3>
                    <h2 style='color: white; margin: 0;'>{}</h2>
                </div>
            """.format(f"{win_rate}%"), unsafe_allow_html=True)
            
        with col3:
            podium_rate = round(stats["Podiums"] / stats["Race Entries"] * 100, 1)
            st.markdown("""
                <div style='background: linear-gradient(145deg, #2E2E2E, #1A1A1A); padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #E10600;'>Podium Rate 🥇</h3>
                    <h2 style='color: white; margin: 0;'>{}</h2>
                </div>
            """.format(f"{podium_rate}%"), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div style='background: linear-gradient(145deg, #2E2E2E, #1A1A1A); padding: 20px; border-radius: 10px; text-align: center;'>
                    <h3 style='color: #E10600;'>Poles ⚡</h3>
                    <h2 style='color: white; margin: 0;'>{}</h2>
                </div>
            """.format(stats["Poles"]), unsafe_allow_html=True)

        # Detailed stats in expandable section
        with st.expander("View Detailed Statistics"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Career Overview")
                st.markdown(f"""
                    - **Seasons**: {stats['Seasons']}
                    - **Race Entries**: {stats['Race Entries']}
                    - **Total Points**: {stats['Points']}
                    - **Fastest Laps**: {stats['Fastest Laps']}
                """)
                
            with col2:
                st.markdown("### Milestone Races")
                st.markdown(f"""
                    - **First Win**: {stats['First Win']}
                    - **Last Win**: {stats['Last Win']}
                    - **Teams**: {', '.join(stats['Teams'])}
                """)

        # Career progression visualization
        st.subheader("Career Progression")
        
        # Example career progression data (customize for each driver)
        if selected_driver == "Lewis Hamilton":
            years = list(range(2007, 2024))
            wins = [4, 5, 2, 3, 3, 4, 1, 11, 10, 9, 9, 11, 11, 11, 8, 0, 0]
            championships = [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        elif selected_driver == "Max Verstappen":
            years = list(range(2015, 2024))
            wins = [0, 1, 2, 2, 3, 2, 10, 15, 19]
            championships = [0, 0, 0, 0, 0, 0, 1, 1, 1]
        else:  # Fernando Alonso
            years = list(range(2001, 2024))
            wins = [0, 0, 1, 0, 7, 7, 4, 2, 2, 5, 1, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            championships = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        # Create progression chart
        fig = go.Figure()
        
        # Add wins bars
        fig.add_trace(go.Bar(
            x=years,
            y=wins,
            name="Wins",
            marker_color='#E10600'
        ))
        
        # Add championship points
        fig.add_trace(go.Scatter(
            x=years,
            y=[c * max(wins) for c in championships],  # Scale championships to match win height
            name="Championships",
            mode='markers',
            marker=dict(
                size=20,
                symbol='star',
                color='gold'
            )
        ))
        
        fig.update_layout(
            title=f"{selected_driver}'s Career Progression",
            xaxis_title="Year",
            yaxis_title="Wins",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0)',
                bordercolor='rgba(255,255,255,0.1)'
            ),
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_performance_analysis(self):
        """Create driver performance analysis section"""
        st.subheader("📊 Performance Analysis")
        
        # Driver selection
        current_drivers = [
            "Max Verstappen", "Lewis Hamilton", "Charles Leclerc",
            "Lando Norris", "Carlos Sainz", "Fernando Alonso"
        ]
        selected_driver = st.selectbox("Select Driver", current_drivers)
        
        # Create performance metrics
        col1, col2, col3 = st.columns(3)
        
        # Sample data - in real app, this would come from API
        metrics = {
            "Max Verstappen": {"Wins": 54, "Poles": 32, "Podiums": 98},
            "Lewis Hamilton": {"Wins": 103, "Poles": 104, "Podiums": 197},
            "Charles Leclerc": {"Wins": 5, "Poles": 23, "Podiums": 28}
        }
        
        driver_metrics = metrics.get(selected_driver, {"Wins": 0, "Poles": 0, "Podiums": 0})
        
        with col1:
            st.metric("Wins 🏆", driver_metrics["Wins"])
        with col2:
            st.metric("Pole Positions ⚡", driver_metrics["Poles"])
        with col3:
            st.metric("Podiums 🥇", driver_metrics["Podiums"])
        
        # Create performance trends chart
        performance_data = {
            "Race": list(range(1, 11)),
            "Position": [2, 1, 1, 3, 1, 2, 1, 1, 2, 1],  # Sample data
            "Points": [18, 25, 25, 15, 25, 18, 25, 25, 18, 25]
        }
        df = pd.DataFrame(performance_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df["Race"],
            y=df["Position"],
            mode='lines+markers',
            name='Position',
            line=dict(color='#E10600', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title=f"{selected_driver}'s Recent Performance Trend",
            yaxis_title="Position",
            yaxis_autorange="reversed",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_head_to_head_comparison(self):
        """Create head-to-head driver comparison section"""
        st.subheader("🤺 Head-to-Head Comparison")
        
        # Driver selection
        col1, col2 = st.columns(2)
        with col1:
            driver1 = st.selectbox("Select First Driver", ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc"])
        with col2:
            driver2 = st.selectbox("Select Second Driver", ["Lewis Hamilton", "Charles Leclerc", "Max Verstappen"])
        
        # Create comparison metrics
        comparison_data = {
            "Metric": ["Qualifying", "Race Pace", "Tire Management", "Wet Weather", "Overtaking"],
            driver1: [95, 98, 90, 95, 92],
            driver2: [92, 90, 95, 88, 85]
        }
        
        df = pd.DataFrame(comparison_data)
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=df[driver1],
            theta=df["Metric"],
            fill='toself',
            name=driver1,
            line_color='#E10600'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=df[driver2],
            theta=df["Metric"],
            fill='toself',
            name=driver2,
            line_color='#3366CC'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def create_driving_style_analysis(self):
        """Create driving style analysis section"""
        st.subheader("🎯 Driving Style Analysis")
        
        # Driver selection
        selected_driver = st.selectbox(
            "Select Driver for Style Analysis",
            ["Max Verstappen", "Lewis Hamilton", "Charles Leclerc", "Fernando Alonso"]
        )
        
        # Driving style metrics
        style_metrics = {
            "Max Verstappen": {
                "Aggressive Overtaking": 95,
                "Defensive Skills": 92,
                "Tire Management": 88,
                "Qualifying Pace": 94,
                "Race Pace": 96,
                "Wet Weather": 95,
                "Technical Feedback": 90,
                "Adaptability": 93
            },
            "Lewis Hamilton": {
                "Aggressive Overtaking": 90,
                "Defensive Skills": 95,
                "Tire Management": 94,
                "Qualifying Pace": 95,
                "Race Pace": 94,
                "Wet Weather": 96,
                "Technical Feedback": 92,
                "Adaptability": 95
            }
        }
        
        # Get default metrics if driver not in dictionary
        metrics = style_metrics.get(selected_driver, {metric: 85 for metric in style_metrics["Max Verstappen"].keys()})
        
        # Create spider chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=selected_driver,
            line_color='#E10600'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Driving style description
        style_descriptions = {
            "Max Verstappen": "Known for his aggressive overtaking and exceptional car control. Verstappen's style is characterized by late braking and precise throttle control.",
            "Lewis Hamilton": "Smooth and precise driving style with excellent tire management. Hamilton excels in both wet conditions and qualifying sessions."
        }
        
        description = style_descriptions.get(
            selected_driver,
            "A balanced driving style combining technical precision with racing instinct."
        )
        
        st.markdown(f"""
            <div style='background-color: #2E2E2E; padding: 20px; border-radius: 10px;'>
                <h4>Driving Style Analysis</h4>
                <p>{description}</p>
            </div>
        """, unsafe_allow_html=True)