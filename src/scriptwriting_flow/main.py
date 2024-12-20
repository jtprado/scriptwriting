# src/scriptwriting_flow/main.py
import os
from typing import List
from pydantic import BaseModel

from crewai.flow.flow import Flow, listen, start

from scriptwriting_flow.crews.research_crew.research_crew import ResearchCrew
from scriptwriting_flow.crews.scriptwriting_crew.scriptwriting_crew import ScriptwritingCrew
from scriptwriting_flow.crews.visual_crew.visual_crew import VisualCrew
from scriptwriting_flow.crews.validation_crew.validation_crew import ContentValidationCrew

from scriptwriting_flow.types import (
    HistoryResearchSeries, 
    HistoryScriptSeries, 
    VisualPromptSeries, 
    ValidationReport
)

class ScriptFlowState(BaseModel):
    topic: str = "Invention of Radio"
    age_range: str = "18-30"
    target_audience: str = "male history enthusiasts who enjoy history told like it happened"
    interests: List[str] = ["history", "science", "technology"]
    research_series: HistoryResearchSeries | None = None
    script_series: HistoryScriptSeries | None = None
    visuals_series: VisualPromptSeries | None = None
    validation_report: ValidationReport | None = None
    iteration_done: bool = False

class ScriptFlow(Flow[ScriptFlowState]):
    initial_state = ScriptFlowState

    @start()
    def perform_research(self):
        # Run research crew and get a HistoryResearchSeries
        research_crew = ResearchCrew().crew()
        research_output = research_crew.kickoff(
            inputs={
                "topic": self.state.topic,
                "age_range": self.state.age_range,
                "target_audience": self.state.target_audience,
                "interests": self.state.interests,
            }
        )
        self.state.research_series = research_output.pydantic
        return research_output

    @listen("perform_research")
    def write_script(self):
        # Pass the research output to the scriptwriting crew
        scriptwriting_crew = ScriptwritingCrew().crew()
        script_output = scriptwriting_crew.kickoff(
            inputs={
                "research_parts": [part.model_dump() for part in self.state.research_series.parts],
                "target_audience": self.state.target_audience,
            }
        )
        self.state.script_series = script_output.pydantic
        return script_output

    @listen("write_script")
    def create_visuals(self):
        # Pass script output to visual crew
        visual_crew = VisualCrew().crew()
        visuals_output = visual_crew.kickoff(
            inputs={
                "scripts_parts": [script.model_dump() for script in self.state.script_series.parts],
                "target_audience": self.state.target_audience,
            }
        )
        self.state.visuals_series = visuals_output.pydantic
        return visuals_output

    @listen("create_visuals")
    def validate_content(self):
        # Pass script and visuals to validation crew
        validation_crew = ContentValidationCrew().crew()
        validation_output = validation_crew.kickoff(
            inputs={
                "scripts_output": [script.model_dump() for script in self.state.script_series.parts],
                "visuals_output": [v.model_dump() for v in self.state.visuals_series.parts],
                "target_audience": self.state.target_audience,
            }
        )
        self.state.validation_report = validation_output.pydantic
        return validation_output

    @listen("validate_content")
    def maybe_improve_script(self):
        # If improvements are suggested and not done yet, rerun script & visuals
        if self.state.validation_report and self.state.validation_report.improvement_suggestions and not self.state.iteration_done:
            self.state.iteration_done = True
            scriptwriting_crew = ScriptwritingCrew().crew()
            script_output = scriptwriting_crew.kickoff(
                inputs={
                    "research_parts": [part.model_dump() for part in self.state.research_series.parts],
                    "target_audience": self.state.target_audience,
                    "critic_feedback": self.state.validation_report.improvement_suggestions,
                }
            )
            self.state.script_series = script_output.pydantic

            visual_crew = VisualCrew().crew()
            visuals_output = visual_crew.kickoff(
                inputs={
                    "scripts_parts": [script.model_dump() for script in self.state.script_series.parts],
                    "target_audience": self.state.target_audience,
                }
            )
            self.state.visuals_series = visuals_output.pydantic
            print("Flow completed after improvements.")
        else:
            print("No further improvements needed. Flow completed.")

def kickoff():
    script_flow = ScriptFlow()
    script_flow.kickoff()

def plot():
    script_flow = ScriptFlow()
    script_flow.plot()

if __name__ == "__main__":
    kickoff()