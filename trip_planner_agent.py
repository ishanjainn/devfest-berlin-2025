#!/usr/bin/env python3
"""
CrewAI Trip Planner Agent
A comprehensive AI-powered trip planning system using multiple specialized agents.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# Try to import search tools, but don't fail if they're not available
try:
    from crewai_tools import SerperDevTool, ScrapeWebsiteTool
    SEARCH_TOOLS_AVAILABLE = True
except ImportError:
    SEARCH_TOOLS_AVAILABLE = False
    print("⚠️ Search tools not available. Install crewai-tools for web search capabilities.")


class TripPlannerCrew:
    """Main trip planner crew orchestrator"""
    
    def __init__(self):
        """Initialize the trip planner with OpenAI model and tools"""
        # Initialize OpenAI model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize tools only if available and properly configured
        self.tools_available = False
        self.search_tool = None
        self.scrape_tool = None
        
        if SEARCH_TOOLS_AVAILABLE:
            # Check if Serper API key is available
            serper_key = os.getenv("SERPER_API_KEY")
            if serper_key:
                try:
                    self.search_tool = SerperDevTool()
                    self.scrape_tool = ScrapeWebsiteTool()
                    self.tools_available = True
                    print("✅ Search tools initialized successfully")
                except Exception as e:
                    print(f"⚠️ Failed to initialize search tools: {e}")
                    print("🔧 Running with LLM knowledge only")
            else:
                print("⚠️ SERPER_API_KEY not found. Set it for web search capabilities:")
                print("   export SERPER_API_KEY='your-serper-api-key'")
                print("🔧 Running with LLM knowledge only")
        else:
            print("🔧 Running with LLM knowledge only")
    
    def create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents for trip planning"""
        
        # Research Agent
        research_tools = []
        if self.tools_available and self.search_tool and self.scrape_tool:
            research_tools = [self.search_tool, self.scrape_tool]
            
        research_agent = Agent(
            role="Travel Research Specialist",
            goal="Research destinations, attractions, and travel logistics to provide comprehensive information",
            backstory="""You are an experienced travel researcher with extensive knowledge of global destinations.
            You excel at finding the best attractions, local customs, weather patterns, and travel requirements
            for any destination. Your research is thorough, accurate, and focuses on providing practical
            information that helps create amazing travel experiences.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=research_tools
        )
        
        # Planning Agent
        planning_agent = Agent(
            role="Travel Itinerary Planner",
            goal="Create detailed, practical, and enjoyable travel itineraries based on research and preferences",
            backstory="""You are a master travel planner who creates perfectly balanced itineraries.
            You understand how to optimize travel time, account for transportation, and balance activities
            with relaxation. Your itineraries are detailed, realistic, and designed to maximize enjoyment
            while minimizing stress and logistics issues.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Budget Analyst Agent
        budget_agent = Agent(
            role="Travel Budget Analyst",
            goal="Analyze costs and provide detailed budget breakdowns for travel plans",
            backstory="""You are a financial expert specializing in travel costs. You have comprehensive
            knowledge of accommodation prices, meal costs, activity fees, and transportation expenses
            across different destinations and travel styles. You provide accurate cost estimates and
            money-saving tips without compromising the travel experience.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Local Experience Agent
        local_tools = []
        if self.tools_available and self.search_tool:
            local_tools = [self.search_tool]
            
        local_agent = Agent(
            role="Local Experience Curator",
            goal="Recommend authentic local experiences, hidden gems, and cultural insights",
            backstory="""You are a cultural expert and local experience curator who knows the hidden gems
            and authentic experiences that make travel memorable. You understand local customs, recommend
            off-the-beaten-path attractions, and suggest ways to connect with local culture and communities.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
            tools=local_tools
        )
        
        return {
            'researcher': research_agent,
            'planner': planning_agent,
            'budget_analyst': budget_agent,
            'local_expert': local_agent
        }
    
    def create_tasks(self, agents: Dict[str, Agent], trip_details: Dict[str, Any]) -> list:
        """Create tasks for the trip planning process"""
        
        # Research Task
        research_task = Task(
            description=f"""
            Research comprehensive information about {trip_details['destination']} for a {trip_details['duration']}-day trip.
            
            Trip Details:
            - Destination: {trip_details['destination']}
            - Duration: {trip_details['duration']} days
            - Travelers: {trip_details['travelers']} people
            - Budget Range: {trip_details['budget']}
            - Travel Dates: {trip_details['dates']}
            - Interests: {', '.join(trip_details['interests'])}
            - Travel Style: {trip_details['travel_style']}
            
            Research Requirements:
            1. Best time to visit and weather conditions
            2. Top attractions and must-see places
            3. Transportation options (airports, local transport)
            4. Accommodation areas and recommendations
            5. Local customs and cultural considerations
            6. Safety information and travel requirements
            7. Currency and payment methods
            8. Language considerations
            
            Provide detailed, practical information that will help create an amazing itinerary.
            """,
            expected_output="A comprehensive research report with practical travel information",
            agent=agents['researcher']
        )
        
        # Local Experiences Task
        local_task = Task(
            description=f"""
            Based on the research findings, identify authentic local experiences and hidden gems in {trip_details['destination']}.
            
            Focus on:
            1. Unique local experiences that match interests: {', '.join(trip_details['interests'])}
            2. Hidden gems and off-the-beaten-path attractions
            3. Local food experiences and must-try dishes
            4. Cultural activities and festivals (if any during travel dates)
            5. Ways to connect with local communities
            6. Authentic shopping opportunities
            7. Local customs and etiquette tips
            
            Prioritize experiences that align with the {trip_details['travel_style']} travel style.
            """,
            expected_output="Curated list of authentic local experiences and cultural insights",
            agent=agents['local_expert'],
            context=[research_task]
        )
        
        # Budget Analysis Task
        budget_task = Task(
            description=f"""
            Create a detailed budget breakdown for the {trip_details['duration']}-day trip to {trip_details['destination']}.
            
            Budget Parameters:
            - Total Budget: {trip_details['budget']}
            - Number of Travelers: {trip_details['travelers']}
            - Travel Style: {trip_details['travel_style']}
            
            Provide cost estimates for:
            1. Flights/Transportation to destination
            2. Accommodation (per night and total)
            3. Local transportation
            4. Meals (breakfast, lunch, dinner)
            5. Activities and attractions
            6. Shopping and souvenirs
            7. Emergency fund (10-15% of budget)
            
            Include:
            - Daily budget breakdown
            - Money-saving tips
            - Alternative options for different budget levels
            - Payment method recommendations
            """,
            expected_output="Detailed budget analysis with daily breakdown and money-saving tips",
            agent=agents['budget_analyst'],
            context=[research_task, local_task]
        )
        
        # Itinerary Planning Task
        planning_task = Task(
            description=f"""
            Create a detailed day-by-day itinerary for the {trip_details['duration']}-day trip to {trip_details['destination']}.
            
            Use information from research, local experiences, and budget analysis to create:
            
            1. Day-by-day schedule with:
               - Morning, afternoon, and evening activities
               - Recommended timing for each activity
               - Transportation between locations
               - Meal recommendations
               
            2. Practical details:
               - Walking distances and travel times
               - Booking requirements for attractions
               - Alternative activities for bad weather
               - Rest periods and flexibility
               
            3. Travel logistics:
               - Airport transfers
               - Hotel check-in/check-out optimization
               - Luggage storage options
               - Emergency contacts and information
            
            Balance must-see attractions with local experiences while staying within budget.
            Consider the travel style: {trip_details['travel_style']} and interests: {', '.join(trip_details['interests'])}.
            """,
            expected_output="Complete day-by-day itinerary with practical details and logistics",
            agent=agents['planner'],
            context=[research_task, local_task, budget_task]
        )
        
        return [research_task, local_task, budget_task, planning_task]
    
    def plan_trip(self, trip_details: Dict[str, Any]) -> str:
        """Execute the trip planning process"""
        
        # Create agents and tasks
        agents = self.create_agents()
        tasks = self.create_tasks(agents, trip_details)
        
        # Create and execute crew
        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        
        # Execute the crew
        print(f"\n🚀 Starting trip planning for {trip_details['destination']}...")
        print(f"📅 Duration: {trip_details['duration']} days")
        print(f"👥 Travelers: {trip_details['travelers']} people")
        print(f"💰 Budget: {trip_details['budget']}")
        print(f"🎯 Interests: {', '.join(trip_details['interests'])}")
        print("\n" + "="*60 + "\n")
        
        result = crew.kickoff()
        
        return result


def get_trip_details() -> Dict[str, Any]:
    """Collect trip details from user input"""
    print("🌟 Welcome to the AI Trip Planner! 🌟")
    print("Let's plan your perfect trip together!\n")
    
    destination = input("🌍 Where would you like to go? (e.g., Paris, France): ").strip()
    
    while True:
        try:
            duration = int(input("📅 How many days is your trip? (e.g., 7): "))
            break
        except ValueError:
            print("Please enter a valid number of days.")
    
    while True:
        try:
            travelers = int(input("👥 How many people are traveling? (e.g., 2): "))
            break
        except ValueError:
            print("Please enter a valid number of travelers.")
    
    budget = input("💰 What's your total budget? (e.g., $3000, €2500): ").strip()
    
    # Get travel dates
    dates = input("📅 When are you traveling? (e.g., March 15-22, 2024): ").strip()
    
    # Get interests
    print("\n🎯 What are your interests? (Enter comma-separated interests)")
    print("Examples: history, food, nightlife, nature, museums, adventure, shopping, culture")
    interests_input = input("Interests: ").strip()
    interests = [interest.strip() for interest in interests_input.split(",")]
    
    # Get travel style
    print("\n✈️ What's your travel style?")
    print("1. Budget (hostels, street food, free activities)")
    print("2. Mid-range (3-star hotels, mix of experiences)")
    print("3. Luxury (high-end accommodations, premium experiences)")
    
    while True:
        style_choice = input("Choose (1-3): ").strip()
        if style_choice == "1":
            travel_style = "budget"
            break
        elif style_choice == "2":
            travel_style = "mid-range"
            break
        elif style_choice == "3":
            travel_style = "luxury"
            break
        else:
            print("Please choose 1, 2, or 3.")
    
    return {
        'destination': destination,
        'duration': duration,
        'travelers': travelers,
        'budget': budget,
        'dates': dates,
        'interests': interests,
        'travel_style': travel_style
    }


def main():
    """Main function to run the trip planner"""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Optional: Show info about Serper API key for web search
    if not os.getenv("SERPER_API_KEY"):
        print("💡 Optional: For enhanced web search capabilities, set SERPER_API_KEY")
        print("   Get a free key at: https://serper.dev/")
        print("   export SERPER_API_KEY='your-serper-api-key'")
        print("   (The agent will work with LLM knowledge only)\n")
    
    try:
        # Get trip details from user
        trip_details = get_trip_details()
        
        # Initialize trip planner
        planner = TripPlannerCrew()
        
        # Plan the trip
        result = planner.plan_trip(trip_details)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trip_plan_{trip_details['destination'].replace(',', '').replace(' ', '_')}_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Trip Plan for {trip_details['destination']}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            f.write(str(result))
        
        print(f"\n✅ Trip planning completed!")
        print(f"📄 Full itinerary saved to: {filename}")
        print(f"\n🎉 Enjoy your trip to {trip_details['destination']}! 🎉")
        
    except KeyboardInterrupt:
        print("\n\n👋 Trip planning cancelled. Safe travels!")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check your API keys and try again.")


if __name__ == "__main__":
    main()
