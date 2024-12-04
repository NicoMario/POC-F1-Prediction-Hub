# app.py
import streamlit as st
from modules.api_module import F1API
from modules.ml_module import F1Model
from modules.historical_wiki import F1HistoricalWiki
from modules.driver_insights import F1DriverInsights
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import plotly.express as px

class F1Dashboard:
    """
    Main dashboard class handling the UI and user interactions.
    Delegates data fetching to F1API and model operations to F1Model.
    """
    def __init__(self):
        self.api = F1API()
        self.model = F1Model()
        self.wiki = F1HistoricalWiki(self.api)
        self.driver_insights = F1DriverInsights(self.api)
        self.initialize_session_state()
        self.setup_theme()

    def setup_theme(self):
        """Configure the dashboard theme and styling for a futuristic look"""
        st.set_page_config(
            page_title="F1 Prediction Hub",
            page_icon="🏎️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Enhanced futuristic styling
        st.markdown("""
            <style>
            /* Main theme */
            .main {
                background-color: #0A0A0A;
                color: #FFFFFF;
            }
            
            /* Sidebar styling */
            .css-1d391kg {
                background-color: #1E1E1E;
            }
            
            /* Headers */
            h1, h2, h3 {
                color: #FFFFFF;
                font-family: 'Racing Sans One', sans-serif;
            }
            
            /* Buttons */
            .stButton>button {
                background: linear-gradient(45deg, #E10600, #FF0000);
                color: white;
                border-radius: 20px;
                border: none;
                padding: 10px 24px;
                box-shadow: 0 4px 15px rgba(225,6,0,0.3);
                transition: all 0.3s ease;
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(225,6,0,0.4);
            }
            
            /* Metrics */
            .stMetric {
                background: linear-gradient(145deg, #2E2E2E, #1A1A1A);
                padding: 20px;
                border-radius: 15px;
                border: 1px solid rgba(225,6,0,0.1);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            /* Cards */
            .custom-card {
                background: linear-gradient(145deg, #2E2E2E, #1A1A1A);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(225,6,0,0.1);
                box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            }
            
            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background-color: #1E1E1E;
                padding: 10px;
                border-radius: 10px;
            }
            
            .stTabs [data-baseweb="tab"] {
                background-color: #2E2E2E;
                border-radius: 8px;
                color: white;
                padding: 8px 16px;
            }
            
            .stTabs [data-baseweb="tab"]:hover {
                background-color: #3E3E3E;
            }
            
            .stTabs [data-baseweb="tab"][aria-selected="true"] {
                background-color: #E10600;
            }
            
            /* Animations */
            @keyframes glow {
                0% { box-shadow: 0 0 5px #E10600; }
                50% { box-shadow: 0 0 20px #E10600; }
                100% { box-shadow: 0 0 5px #E10600; }
            }
            
            .glow-effect {
                animation: glow 2s infinite;
            }
            </style>
            
            <!-- Racing Sans One font -->
            <link href="https://fonts.googleapis.com/css2?family=Racing+Sans+One&display=swap" rel="stylesheet">
            """, unsafe_allow_html=True)

    def create_header(self):
        """Create animated dashboard header"""
        st.markdown("""
            <div style='text-align: center; padding: 20px;'>
                <h1 style='font-size: 3em; margin-bottom: 0;'>🏎️ F1 PREDICTION HUB</h1>
                <p style='color: #E10600; font-size: 1.2em;'>Powered by Machine Learning</p>
            </div>
        """, unsafe_allow_html=True)

    def initialize_session_state(self):
        """Initialize all required session state variables"""
        default_state = {
            'model_trained': False,
            'training_data': None,
            'data_cache': {},
            'metrics': None,
            'selected_features': None,
            'selected_constructor': None,
            'current_round': 1,
            'model_state': None,  # Add model state storage
            'session_initialized': True
        }
        
        for key, value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def create_sidebar(self):
            """Create enhanced sidebar with fixed race countdown"""
            with st.sidebar:
                # F1 Logo with glow effect
                st.markdown("""
                    <div class='glow-effect' style='text-align: center; padding: 20px;'>
                        <img src='https://www.formula1.com/etc/designs/fom-website/images/f1_logo.svg'
                            style='width: 200px;'>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Next race countdown with proper calculation
                next_race = self.api.get_next_race()
                current_time = datetime.now()
                
                # Calculate time difference components
                time_diff = next_race['date'] - current_time
                days = time_diff.days
                hours = time_diff.seconds // 3600
                minutes = (time_diff.seconds % 3600) // 60
                
                st.markdown(f"""
                    <div style='text-align: center;'>
                        <h3>NEXT RACE</h3>
                        <h2>{next_race['name']}</h2>
                        <div class='glow-effect' style='background: #2E2E2E; padding: 20px; border-radius: 10px;'>
                            <h1 style='color: #E10600; font-size: 3em; margin: 0;'>{days}d {hours}h</h1>
                            <p style='margin: 0;'>UNTIL LIGHTS OUT</p>
                        </div>
                        <p style='font-size: 0.9em; margin-top: 10px;'>🏁 {next_race['circuit']}</p>
                        <p style='font-size: 0.8em; color: #888;'>
                            {next_race['date'].strftime('%d %B %Y')}<br>
                            {next_race['date'].strftime('%H:%M')} Local Time
                        </p>
                    </div>
                """, unsafe_allow_html=True)


    def display_model_stats(self):
        """Display current model statistics"""
        st.subheader("Model Statistics")
        metrics = st.session_state.metrics
        if metrics:
            stats = {
                "Data Points": metrics['data_points'],
                "Training Score": f"{metrics['training_score']:.2%}",
                "Test Score": f"{metrics['test_score']:.2%}",
                "Features Used": len(st.session_state.selected_features or [])
            }
            st.json(stats)

    def create_training_section(self):
        """Create and manage the model training interface"""
        st.header("🎯 Model Training Configuration")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            years_range = st.slider(
                "Select year range for training",
                2000, 2024, (2015, 2023)
            )
            
            available_gps = self.api.get_available_grand_prix()
            selected_gps = st.multiselect(
                "Select specific Grand Prix events (leave empty for all)",
                available_gps,
                default=[]
            )
            
            progress_text = st.empty()
            progress_bar = st.progress(0)

        with col2:
            st.subheader("Feature Selection")
            features = {
                "Grid Position": True,
                "Constructor": True,
                "Circuit": True,
                "Round": True
            }
            
            selected_features = {k: st.checkbox(k, value=v) for k, v in features.items()}

        if st.button("Train Model", key="train_button"):
            try:
                # Collect training data
                training_data = pd.DataFrame()
                years = list(range(years_range[0], years_range[1] + 1))
                
                for i, year in enumerate(years):
                    progress_text.text(f"Fetching data for {year}...")
                    progress_bar.progress((i + 1) / len(years))
                    
                    year_data = self.api.get_race_results(
                        year=year,
                        grand_prix=selected_gps if selected_gps else None
                    )
                    
                    if not year_data.empty:
                        training_data = pd.concat([training_data, year_data], ignore_index=True)
                
                if training_data.empty:
                    st.error("No training data collected. Please try different years or Grand Prix selections.")
                    return
                
                st.info(f"Collected {len(training_data)} race results from {len(years)} seasons")
                
                # Train model
                with st.spinner("Training model..."):
                    metrics = self.model.train(training_data, selected_features)
                    
                    # Update session state
                    st.session_state.model_trained = True
                    st.session_state.metrics = metrics
                    st.session_state.training_data = training_data
                    st.session_state.selected_features = selected_features
                    st.session_state.model_state = self.model.get_model_state()  # Store model state
                
                st.success("🎉 Model trained successfully!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Training Score", f"{metrics['training_score']:.2%}")
                col2.metric("Test Score", f"{metrics['test_score']:.2%}")
                col3.metric("Data Points", metrics['data_points'])
                
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                st.session_state.model_trained = False
            finally:
                progress_bar.empty()
                progress_text.empty()

    def get_training_parameters(self):
        """Collect all training parameters from user input"""
        # Year range selection
        years_range = st.slider(
            "Select year range for training",
            2000, 2024, (2020, 2024)
        )
        
        # Grand Prix selection
        available_gps = self.api.get_available_grand_prix()
        selected_gps = st.multiselect(
            "Select specific Grand Prix events",
            available_gps,
            default=available_gps[:3]
        )
        
        # Additional parameters
        params = {
            'years': list(range(years_range[0], years_range[1] + 1)),
            'grand_prix': selected_gps,
            'weather': st.multiselect(
                "Filter by weather conditions",
                ["Dry", "Wet", "Mixed"],
                default=["Dry", "Wet"]
            ),
            'track_temp_range': st.slider(
                "Track Temperature Range (°C)",
                0, 60, (20, 40)
            ),
            'include_sprint_races': st.checkbox("Include Sprint Races", value=True)
        }
        
        return params

    def get_feature_selection(self):
        """Manage feature selection for model training"""
        st.subheader("Feature Selection")
        features = {
            "Grid Position": True,
            "Previous Race Result": True,
            "Practice Times": False,
            "Qualifying Performance": True,
            "Weather Impact": True,
            "Track Temperature": False,
            "Tire Strategy": True,
            "Historical Performance": True,
            "Driver Experience": True
        }
        
        return {k: st.checkbox(k, value=v) for k, v in features.items()}

    def handle_model_training(self, params, features):
        """Handle the model training process"""
        progress_bar = st.progress(0)
        
        try:
            # Collect training data
            training_data = self.collect_training_data(params, progress_bar)
            
            if training_data.empty:
                st.error("No training data collected. Please adjust your selection criteria.")
                return
            
            # Store selected features
            st.session_state.selected_features = features
            
            # Train model
            with st.spinner("Training model with collected data..."):
                metrics = self.model.train(training_data, features)
                st.session_state.model_trained = True
                st.session_state.metrics = metrics
                st.session_state.training_data = training_data
            
            self.display_training_results(metrics)
            
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
        finally:
            progress_bar.empty()

    def collect_training_data(self, params, progress_bar):
        """Collect and process training data based on parameters"""
        training_data = pd.DataFrame()
        
        for i, year in enumerate(params['years']):
            year_data = self.api.get_race_results(
                year,
                grand_prix=params['grand_prix'],
                weather=params['weather'],
                include_sprint=params['include_sprint_races']
            )
            
            if not year_data.empty:
                training_data = pd.concat([training_data, year_data])
            
            progress_bar.progress((i + 1) / len(params['years']))
        
        return training_data

    def display_training_results(self, metrics):
        """Display the results of model training"""
        st.success("🎉 Advanced model trained successfully!")
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Training Score", f"{metrics['training_score']:.2%}")
        col2.metric("Test Score", f"{metrics['test_score']:.2%}")
        col3.metric("Data Points", metrics['data_points'])
        col4.metric("Features Used", len(st.session_state.selected_features))

    def create_prediction_section(self):
        """Create and manage the prediction interface"""
        st.header("🔮 Race Prediction Engine")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first to make predictions")
            return
            
        # Restore model state if necessary
        if st.session_state.model_state and not self.model.is_trained:
            self.model.set_model_state(st.session_state.model_state)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            grid_position = st.number_input("Starting Grid Position", 1, 20, 5)
            selected_track = st.selectbox(
                "Select Track",
                self.api.get_available_grand_prix()
            )
            selected_constructor = st.selectbox(
                "Select Constructor",
                ["Mercedes", "Red Bull", "Ferrari", "McLaren", "Alpine", "AlphaTauri", 
                 "Aston Martin", "Williams", "Alfa Romeo", "Haas"]
            )
        
        with col2:
            weather = st.select_slider(
                "Weather Conditions",
                options=["Dry", "Mixed", "Wet"],
                value="Dry"
            )
            tire_strategy = st.selectbox(
                "Tire Strategy",
                ["Soft-Medium", "Medium-Hard", "Soft-Hard", "Three-Stop"]
            )
            track_temp = st.slider("Track Temperature (°C)", 20, 50, 30)
        
        if st.button("Generate Prediction", key="predict_button"):
            try:
                prediction_input = {
                    'grid_position': grid_position,
                    'track': selected_track,
                    'constructor': selected_constructor,
                    'weather': weather,
                    'tire_strategy': tire_strategy,
                    'track_temp': track_temp
                }
                
                prediction = self.model.predict(prediction_input)
                
                with col3:
                    st.markdown(
                        f"""
                        <div style='background-color: #2E2E2E; padding: 20px; border-radius: 10px;'>
                            <h2 style='color: #E10600'>Race Prediction</h2>
                            <h3>Predicted Finish: P{prediction['predicted_position']}</h3>
                            <p>Confidence: {prediction['confidence']:.0%}</p>
                            <p>Expected Positions Gained: {grid_position - prediction['predicted_position']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                self.create_visualization_section(grid_position, prediction['predicted_position'])
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please ensure you have trained the model with appropriate data first.")

    def get_prediction_parameters(self):
        """Collect prediction parameters from user input"""
        return {
            'grid_position': st.number_input("Starting Grid Position", 1, 20, 5),
            'track': st.selectbox(
                "Select Track",
                self.api.get_available_grand_prix()
            )
        }

    def get_race_conditions(self):
        """Collect race condition parameters"""
        return {
            'weather': st.select_slider(
                "Weather Conditions",
                options=["Dry", "Mixed", "Wet"],
                value="Dry"
            ),
            'tire_strategy': st.selectbox(
                "Tire Strategy",
                ["Soft-Medium", "Medium-Hard", "Soft-Hard", "Three-Stop"]
            ),
            'track_temp': st.slider("Track Temperature (°C)", 20, 50, 30)
        }

    def handle_prediction(self, params, conditions, display_col):
        """Handle the prediction process and display results"""
        try:
            if not hasattr(self.model, 'model') or self.model.model is None:
                raise ValueError("Please train the model first")
                
            # Combine parameters and conditions
            prediction_input = {
                'grid_position': params['grid_position'],
                'track': params['track'],
                'constructor': st.session_state.get('selected_constructor', 'Unknown'),
                'round': st.session_state.get('current_round', 1),
                'weather': conditions['weather'],
                'tire_strategy': conditions['tire_strategy'],
                'track_temp': conditions['track_temp']
            }
            
            # Make prediction
            prediction = self.model.predict(prediction_input)
            self.display_prediction_results(prediction, params, display_col)
            self.create_visualization_section(params['grid_position'], prediction['predicted_position'])
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.error("Please ensure you have trained the model with appropriate data first.")

    def display_prediction_results(self, prediction, params, col):
        """Display prediction results"""
        with col:
            st.markdown(
                f"""
                <div style='background-color: #2E2E2E; padding: 20px; border-radius: 10px;'>
                    <h2 style='color: #E10600'>Race Prediction</h2>
                    <h3>Predicted Finish: P{prediction['predicted_position']}</h3>
                    <p>Confidence: {prediction['confidence']:.0%}</p>
                    <p>Expected Positions Gained: {params['grid_position'] - prediction['predicted_position']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    def create_visualization_section(self, grid_position, predicted_position):
        """Create and display prediction visualizations"""
        st.subheader("📊 Prediction Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_position_change(grid_position, predicted_position)
        
        with col2:
            self.plot_historical_heatmap()

    def plot_position_change(self, grid_position, predicted_position):
        """Create position change visualization"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=['Start', 'Predicted Finish'],
            y=[grid_position, predicted_position],
            mode='lines+markers',
            name='Position Change',
            line=dict(color='#E10600', width=4),
            marker=dict(size=12)
        ))
        
        fig.update_layout(
            title="Predicted Position Change",
            yaxis_title="Position",
            yaxis_autorange="reversed",
            plot_bgcolor='#2E2E2E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def plot_historical_heatmap(self):
        """Create historical performance heatmap"""
        if st.session_state.training_data is not None:
            historical_data = st.session_state.training_data
            heatmap_data = historical_data.groupby(['grid', 'position']).size().unstack(fill_value=0)
            
            fig = px.imshow(
                heatmap_data,
                labels=dict(x="Finish Position", y="Grid Position", color="Frequency"),
                color_continuous_scale="Reds"
            )
            
            fig.update_layout(
                title="Historical Grid vs Finish Position Heatmap",
                plot_bgcolor='#2E2E2E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def run(self):
        """Run the enhanced dashboard application"""
        self.create_sidebar()
        self.create_header()
        
        # Create main tabs
        tabs = st.tabs([
            "🎯 Race Predictions",
            "📚 Historical Wiki",
            "👨‍🚀 Driver Insights"
        ])
        
        with tabs[0]:
            # Original prediction functionality
            self.create_training_section()
            st.markdown("---")
            self.create_prediction_section()
            
        with tabs[1]:
            # Historical wiki section
            self.wiki.create_historical_section()
            
        with tabs[2]:
            # Driver insights section
            self.driver_insights.create_driver_insights_section()

if __name__ == "__main__":
    dashboard = F1Dashboard()
    dashboard.run()