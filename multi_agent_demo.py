# Manage warnings
import warnings

warnings.filterwarnings("ignore")
from dotenv import load_dotenv

# Load Open AI Api Key
import os
load_dotenv('.env')
openai_api_key: str = os.getenv('OPENAI_KEY')

# Import CrewAI
from crewai import Agent, Task, Crew

os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o'
os.environ['OPENAI_API_KEY'] = openai_api_key

# ----------------------------------------------------------------------------------------------------------------------
# 1. Create Agents objects
planner = Agent(
    role='Content Planner',
    goal='Plan engaging and factually accurate content on {topic}',
    backstory="You are working on planning a blog post article about the topic: {topic}. "
              "You collect information that helps the audience learn something and make informed decisions. "
              "Your work is the basis for the Content Writer an article on this topic.",
    allow_delegation=False,
    verbose=True,
    tools=['ScrapeElementFromWebsiteTool', 'WebsiteSearchTool', 'YoutubeChannelSearchTool', 'JSONSearchTool', 'DirectorySearchTool']
)

writer = Agent(
    role='Content Writer',
    goal='Write insightful and factually accurate opinion piece about the topic: {topic}',
    backstory="You are working on a writing a new opinion piece about the topic: {topic}. "
              "You base your writing on the work of the Content Planner, who provides an outline and relevant context"
              "about the topic. "
              "You follow the main objectives and directions of the outline, as provided by the Content Planner."
              "You acknowledge in your opinion piece when your statements are opinions as opposed to objective "
              "statements.",
    allow_delegation=False,
    verbose=True,
)

editor = Agent(
    role='Editor',
    goal='Edit a given blog post to align with the writing style of the organization',
    backstory="You are an editor who receives a blog post from the Content Writer. "
              "Your goal is to review the blog post to ensure that it follows journalistic best practices, provides"
              "balanced viewpoints when providing opinions or assertions, and also avoids major controversial topics "
              "or opinions when possible",
    allow_delegation=False,
    verbose=True,
)

# 2. Creating tasks for the agent
plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, and noteworthy news on {topic}.\n"
        "2. Identify the target audience, considering their interests and pain points.\n"
        "3. Develop a detailed content outline including an instruction, key points, and a call to an actions.\n"
        "4. Include SEO keywords and relevant data or sources."
    ),
    expected_output="A comprehensive content plan document with an outline, audience analysis, SEO keywords, and "
                    "resources.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use the content plan to craft a compelling blog post on {topic}.\n"
        "2. Incorporate SEO keywords naturally.\n"
        "3. Sections/Subtitles are properly named in an engaging manner.\n"
        "4. Ensure the post is structured with an engaging instruction, insightful body, and summarizing conclusion.\n"
        "5. Proofread for grammatical errors and alignment with the brand voice."
    ),
    expected_output="A well-written blog post in markdown format, ready for publication, each section should have 1 or "
                    "2 paragraphs.",
    agent=writer,
)

edit = Task(
    description=(
        "Proofread the given blogpost for grammatical errors and alignment with the brand voice."
    ),
    expected_output="A well-written blog post in markdown format, ready for publication, each section should have 1 or "
                    "2 paragraphs.",
    agent=editor,
)

# 3. Connect everything together!
# Pass the tasks to the performed by those agents (the task will be performed sequentially, they are dependent on each
# other), so the order of the task in the list matters.
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True,
)

# 4. Run the crew!
TOPIC: str = "Web3 SEO"
result = crew.kickoff(
    inputs={
        'topic': TOPIC
    }
)