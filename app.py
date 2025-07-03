import streamlit as st
import os
from datetime import datetime, timedelta
from gmail_reader import authenticate_gmail, get_latest_emails, get_email_by_id
from nlp_processor import EmailProcessor

# Initialize session state
if 'gmail_service' not in st.session_state:
    st.session_state.gmail_service = None
if 'email_processor' not in st.session_state:
    st.session_state.email_processor = EmailProcessor()
if 'processed_emails' not in st.session_state:
    st.session_state.processed_emails = []

def initialize_gmail():
    """Initialize Gmail service if credentials are available"""
    try:
        if os.path.exists('credentials.json'):
            st.session_state.gmail_service = authenticate_gmail()
            return True
        return False
    except Exception as e:
        st.error(f"Error initializing Gmail: {e}")
        return False

def format_date(date_str):
    """Format the email date string"""
    try:
        date_obj = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
        return date_obj.strftime('%Y-%m-%d %H:%M')
    except:
        return date_str

# Page configuration
st.set_page_config(
    page_title="Email Assistant",
    page_icon="📧",
    layout="wide"
)

# Main UI
st.title("📧 Personal Email Assistant")

# Sidebar
with st.sidebar:
    st.header("Controls")
    if st.button("🔄 Refresh Emails"):
        if st.session_state.gmail_service:
            with st.spinner("Fetching emails..."):
                emails = get_latest_emails(st.session_state.gmail_service)
                st.session_state.processed_emails = [
                    st.session_state.email_processor.process_email(email)
                    for email in emails
                ]
            st.success(f"Fetched {len(emails)} emails!")
        else:
            st.error("Gmail service not initialized. Please check your credentials.")

    st.sidebar.markdown("---")
    
    # Email count selector
    email_count = st.sidebar.slider("Number of emails to fetch", 10, 100, 50)
    
    # Filter options
    st.sidebar.markdown("### Filter Options")
    filter_sender = st.sidebar.text_input("Filter by sender")
    filter_subject = st.sidebar.text_input("Filter by subject")
    
    # Date range filter
    st.sidebar.markdown("### Date Range")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    date_range = st.sidebar.date_input(
        "Select date range",
        value=(start_date, end_date),
        max_value=end_date
    )

    st.markdown("---")
    st.markdown("""
    ### How to use:
    1. Make sure you have `credentials.json` in the project directory
    2. Click 'Refresh Emails' to fetch latest emails
    3. View email summaries and analysis
    4. Use filters to find specific emails
    """)

# Main content
if not st.session_state.gmail_service:
    if os.path.exists('credentials.json'):
        if initialize_gmail():
            st.success("Gmail service initialized successfully!")
        else:
            st.error("Failed to initialize Gmail service. Please check your credentials.")
    else:
        st.error("""
        Please add your Gmail API credentials:
        1. Go to Google Cloud Console
        2. Create a project and enable Gmail API
        3. Create OAuth credentials (Desktop app)
        4. Download and save as 'credentials.json' in the project directory
        """)
else:
    # Email display
    if not st.session_state.processed_emails:
        st.info("Click 'Refresh Emails' to fetch your latest emails!")
    else:
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📥 Inbox", "📊 Analytics", "🔍 Search"])
        
        # Filter emails based on criteria
        filtered_emails = st.session_state.processed_emails
        
        if filter_sender:
            filtered_emails = [
                email for email in filtered_emails 
                if filter_sender.lower() in email['original_email']['sender'].lower()
            ]
            
        if filter_subject:
            filtered_emails = [
                email for email in filtered_emails 
                if filter_subject.lower() in email['original_email']['subject'].lower()
            ]
        
        with tab1:
            # Email counter
            st.markdown(f"### Showing {len(filtered_emails)} emails")
            
            for email in filtered_emails:
                with st.expander(f"📧 {email['original_email']['subject']}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**From:** {email['original_email']['sender']}")
                        st.markdown(f"**Date:** {format_date(email['original_email']['date'])}")
                        
                        st.markdown("### Summary")
                        st.write(email['summary'])
                        
                        st.markdown("### Key Points")
                        for point in email['key_points']:
                            st.markdown(f"• {point}")
                            
                        if st.button("Show Original", key=email['original_email']['id']):
                            st.markdown("### Original Email")
                            st.text_area("", email['original_email']['body'], height=200)
                    
                    with col2:
                        sentiment = email['sentiment']
                        st.markdown("### Sentiment Analysis")
                        
                        # Display sentiment with emoji and color
                        if sentiment['sentiment'] == "POSITIVE":
                            emoji = "😊"
                            color = "green"
                        elif sentiment['sentiment'] == "NEGATIVE":
                            emoji = "😔"
                            color = "red"
                        else:
                            emoji = "😐"
                            color = "gray"
                            
                        st.markdown(f"**{sentiment['sentiment']}** {emoji}")
                        
                        # Create a progress bar for confidence
                        st.progress(sentiment['confidence'])
                        st.caption(f"Confidence: {sentiment['confidence']*100:.1f}%")
        
        with tab2:
            # Enhanced analytics
            total_emails = len(filtered_emails)
            
            # Sentiment distribution
            sentiments = [e['sentiment']['sentiment'] for e in filtered_emails]
            positive_count = sum(1 for s in sentiments if s == "POSITIVE")
            negative_count = sum(1 for s in sentiments if s == "NEGATIVE")
            neutral_count = total_emails - positive_count - negative_count
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Emails", total_emails)
                st.metric("Positive Emails", positive_count)
                st.metric("Negative Emails", negative_count)
                st.metric("Neutral Emails", neutral_count)
            
            with col2:
                # Calculate sentiment distribution
                if total_emails > 0:
                    sentiment_dist = {
                        "Positive": (positive_count / total_emails) * 100,
                        "Negative": (negative_count / total_emails) * 100,
                        "Neutral": (neutral_count / total_emails) * 100
                    }
                    
                    st.markdown("### Sentiment Distribution")
                    st.bar_chart(sentiment_dist)
            
            # Add sender analysis
            st.markdown("### Top Senders")
            sender_counts = {}
            for email in filtered_emails:
                sender = email['original_email']['sender']
                sender_counts[sender] = sender_counts.get(sender, 0) + 1
            
            # Sort and display top senders
            top_senders = dict(sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            st.bar_chart(top_senders)
        
        with tab3:
            # Advanced search
            st.markdown("### Advanced Search")
            search_query = st.text_input("Search in email content")
            
            if search_query:
                search_results = [
                    email for email in filtered_emails
                    if search_query.lower() in email['original_email']['body'].lower()
                    or search_query.lower() in email['original_email']['subject'].lower()
                ]
                
                st.markdown(f"Found {len(search_results)} matching emails")
                
                for email in search_results:
                    with st.expander(f"📧 {email['original_email']['subject']}"):
                        st.markdown(f"**From:** {email['original_email']['sender']}")
                        st.markdown(f"**Date:** {format_date(email['original_email']['date'])}")
                        st.markdown("### Summary")
                        st.write(email['summary']) 