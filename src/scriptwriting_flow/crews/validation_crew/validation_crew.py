# src/scriptwriting_flow/crews/validation_crew/validation_crew.py
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from scriptwriting_flow.types import ValidationReport

@CrewBase
class ContentValidationCrew():
    """Content Validation Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def content_critic(self) -> Agent:
        return Agent(
            config=self.agents_config["content_critic"],
            verbose=True
        )

    @task
    def validate_content(self) -> Task:
        # No context; we get inputs from main flow
        return Task(
            config=self.tasks_config["content_validation"],
            output_pydantic=ValidationReport
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=[self.validate_content()],
            process=Process.sequential,
            verbose=True
        )