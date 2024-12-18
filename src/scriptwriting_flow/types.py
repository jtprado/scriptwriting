from pydantic import BaseModel
from typing import List, Dict, Optional

# Deep research output
class HistoryResearchReport(BaseModel):
    summary: str
    key_figures: List[str]
    timeline: List[str]
    societal_impact: str
    sources: List[Dict[str, str]]

# New: 3-part structured research
class HistoryResearchPart(BaseModel):
    summary: str
    key_figures: List[str]
    timeline: List[str]
    societal_impact: str
    sources: List[Dict[str, str]]

class HistoryResearchSeries(BaseModel):
    parts: List[HistoryResearchPart]

# Script generation output
class ScriptSegment(BaseModel):
    timestamp: str
    text: str

class HistoryScript(BaseModel):
    title: str
    hook: str
    segments: List[ScriptSegment]
    conclusion: str

# New: series of 3 scripts
class HistoryScriptSeries(BaseModel):
    parts: List[HistoryScript]

# Generate visuals output
class VisualPrompt(BaseModel):
    timestamp: str
    description: str
    composition: Dict[str, str]
    style: str

class VisualPromptSet(BaseModel):
    prompts: List[VisualPrompt]

# New: series of 3 VisualPromptSets
class VisualPromptSeries(BaseModel):
    parts: List[VisualPromptSet]

# Content validation output
class ValidationReport(BaseModel):
    improvement_suggestions: Optional[str] = None