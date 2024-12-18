# research_crew.py (updated according to docs)
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

from langchain_openai import ChatOpenAI

from scriptwriting_flow.types import HistoryResearchReport, HistoryResearchSeries

@CrewBase
class ResearchCrew():
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    # llm = ChatOpenAI(model="gpt-4o-mini")

    @agent
    def historian(self) -> Agent:
        search_tool = SerperDevTool()
        return Agent(
            config=self.agents_config['historian'],
            tools=[search_tool],
            verbose=True,
            allow_delegation=False
        )

    @agent
    def splitter(self) -> Agent:
        return Agent(
            config=self.agents_config['splitter'],
            verbose=True,
            allow_delegation=False
        )

    @task
    def deep_research(self) -> Task:
        conf = self.tasks_config['deep_research']
        return Task(
            config=self.tasks_config['deep_research'],
            output_pydantic=HistoryResearchReport
        )

    @task
    def split_research(self) -> Task:
        return Task(
            config=self.tasks_config["split_research"],    
            output_pydantic=HistoryResearchSeries,
            context=[HistoryResearchReport],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # agents discovered from @agent decorated methods
            tasks=self.tasks,    # tasks discovered from @task decorated methods
            process=Process.sequential,
            verbose=True,
        )