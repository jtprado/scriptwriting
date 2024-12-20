# src/scriptwriting_flow/crews/research_crew/research_crew.py
import os
import dotenv

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, BraveSearchTool, ScrapeWebsiteTool
from langchain_openai import ChatOpenAI

from scriptwriting_flow.types import HistoryResearchReport, HistoryResearchSeries

dotenv.load_dotenv()

llm = LLM(
    model="anthropic/claude-3-5-sonnet-20241022",
    temperature=0
)


@CrewBase
class ResearchCrew():
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def historian(self) -> Agent:
        # search_tool = SerperDevTool()
        brave_search_tool = BraveSearchTool()
        scrape_rool = ScrapeWebsiteTool()
        return Agent(
            config=self.agents_config['historian'],
            tools=[brave_search_tool, scrape_rool],
            verbose=True,
            llm=self.llm,
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
        return Task(
            config=self.tasks_config['deep_research'],
            output_pydantic=HistoryResearchReport
        )

    @task
    def split_research(self) -> Task:
        # This task depends on deep_research's output
        return Task(
            config=self.tasks_config["split_research"],
            output_pydantic=HistoryResearchSeries,
            context=[self.deep_research()]  # linking the first task's result
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[self.deep_research(), self.split_research()],
            process=Process.sequential,
            verbose=True,
        )
