"""
Hook Generator using CrewAI
============================

Two-agent system for generating engaging hooks from podcast transcriptions:
1. Hook Generator Agent: Creates 8 hook variations with different tones
2. Hook Evaluator Agent: Scores and selects the best hook, identifies SUBJECT/OBJECT

Tones: sarcastic, dismissive, teasing, contrarian, skeptical, faux-naive
"""

import os
import json
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from dotenv import load_dotenv

if TYPE_CHECKING:
    import numpy as np

load_dotenv()


@dataclass
class HookResult:
    """Result from hook generation pipeline."""
    hook_text: str
    subject_word: Optional[str]  # Word to highlight in YELLOW
    object_word: Optional[str]   # Word to highlight in PURPLE
    scores: Dict[str, float]     # Evaluation scores
    reasoning: str               # Why this hook was selected


class HookGeneratorCrew:
    """
    CrewAI-based hook generation system with two agents:
    1. Generator: Creates 8 hook variations
    2. Evaluator: Scores and selects the best hook
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        # Set up environment for CrewAI/LiteLLM to use Gemini
        # LiteLLM expects GEMINI_API_KEY for gemini/ prefixed models
        os.environ["GEMINI_API_KEY"] = self.api_key
        os.environ["GOOGLE_API_KEY"] = self.api_key
        
        # Import CrewAI components
        try:
            from crewai import Agent, Task, Crew, Process
            self.Agent = Agent
            self.Task = Task
            self.Crew = Crew
            self.Process = Process
        except ImportError:
            raise ImportError("Please install crewai: pip install crewai crewai-tools")
    
    def _create_generator_agent(self):
        """Create the hook generator agent using Gemini 2.5 Pro."""
        return self.Agent(
            role="Viral Hook Writer",
            goal="Create 8 provocative, scroll-stopping hooks that trigger emotional responses",
            backstory="""You are a master of viral content and psychological triggers. 
You understand what makes people stop scrolling - controversy, emotional charge, 
and statements that demand a response. You specialize in hooks that are:
- Sarcastic: Dripping with irony that makes people want to correct you
- Dismissive: Casually brushing off something important
- Teasing: Dangling information that creates curiosity gaps
- Contrarian: Going against popular opinion
- Skeptical: Questioning widely-held beliefs
- Faux-naive: Pretending not to understand obvious things

Your hooks are always under 12 words and designed to maximize replies and engagement.""",
            llm="gemini/gemini-2.5-pro",
            verbose=True,
            allow_delegation=False
        )
    
    def _create_evaluator_agent(self):
        """Create the hook evaluator agent using Gemini 2.0 Flash."""
        return self.Agent(
            role="Engagement Analyst",
            goal="Evaluate hooks and identify the most viral-worthy option with SUBJECT/OBJECT highlights",
            backstory="""You are an expert in social media psychology and engagement metrics.
You analyze hooks based on their potential to:
1. Trigger emotional responses (anger, curiosity, disbelief)
2. Generate controversy and debate
3. Use negativity bias effectively (people engage more with negative content)
4. Remain defensible (provocative but not bannable)

You also identify the KEY SUBJECT (main topic - highlighted YELLOW) and 
KEY OBJECT (what's being said about it - highlighted PURPLE) in each hook
to maximize visual impact and comprehension.""",
            llm="gemini/gemini-3-pro-preview",
            verbose=True,
            allow_delegation=False
        )
    
    def _create_generation_task(self, agent, transcript: str):
        """Create the hook generation task."""
        return self.Task(
            description=f"""Based on this podcast transcript, create exactly 8 hook variations.
Each hook MUST be 12 words or fewer.

TRANSCRIPT:
{transcript}

Create hooks using these tones (at least one of each, you can repeat if needed):
1. SARCASTIC - Dripping with irony
2. DISMISSIVE - Casually brush off something important  
3. TEASING - Create a curiosity gap
4. CONTRARIAN - Go against the grain
5. SKEPTICAL - Question the obvious
6. FAUX-NAIVE - Pretend confusion about something clear

RULES:
- Each hook MUST be under 12 words
- Make them controversial enough to trigger replies
- They should relate to the transcript content
- Number each hook 1-8

Output format:
1. [TONE]: "Hook text here"
2. [TONE]: "Hook text here"
... etc for all 8 hooks""",
            expected_output="""A numbered list of exactly 8 hooks, each labeled with its tone type,
each under 12 words, designed to maximize engagement and replies.""",
            agent=agent
        )
    
    def _create_evaluation_task(self, agent, generation_task):
        """Create the hook evaluation task."""
        return self.Task(
            description="""Evaluate all 8 hooks from the previous task.

Score each hook on a scale of 1-10 for:
1. EMOTIONAL_CHARGE: How strongly will this trigger an emotional response?
2. CONTROVERSY_POTENTIAL: How likely to generate debate/arguments?
3. NEGATIVITY_BIAS: How effectively does it use negativity to drive engagement?
4. DEFENSIBILITY: Can you defend posting this? (10 = perfectly defensible, 1 = likely to get banned)

Then select the BEST hook (highest combined score with defensibility >= 7).

For the winning hook, identify:
- SUBJECT: The main topic word (to highlight in YELLOW)
- OBJECT: The key descriptor/action word (to highlight in PURPLE)

OUTPUT YOUR RESPONSE AS VALID JSON with this exact structure:
{
    "hooks_evaluated": [
        {"number": 1, "hook": "hook text", "emotional": 8, "controversy": 7, "negativity": 6, "defensibility": 9, "total": 30},
        ... for all 8 hooks
    ],
    "winner": {
        "number": 1,
        "hook": "the winning hook text",
        "subject_word": "TOPIC",
        "object_word": "DESCRIPTOR",
        "reasoning": "Why this hook will perform best"
    }
}

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no extra text.""",
            expected_output="""A JSON object containing evaluation scores for all 8 hooks
and the winning hook with SUBJECT and OBJECT words identified.""",
            agent=agent,
            context=[generation_task]
        )
    
    def generate_hook(self, transcript: str) -> HookResult:
        """
        Generate the best hook for a given transcript.
        
        Args:
            transcript: The podcast transcript text
            
        Returns:
            HookResult with the winning hook and highlights
        """
        print("[HOOK] Initializing CrewAI agents...")
        
        # Create agents
        generator = self._create_generator_agent()
        evaluator = self._create_evaluator_agent()
        
        # Create tasks
        generation_task = self._create_generation_task(generator, transcript)
        evaluation_task = self._create_evaluation_task(evaluator, generation_task)
        
        # Create crew
        crew = self.Crew(
            agents=[generator, evaluator],
            tasks=[generation_task, evaluation_task],
            process=self.Process.sequential,
            verbose=True
        )
        
        print("[HOOK] Running hook generation crew...")
        
        # Execute crew
        result = crew.kickoff()
        
        # Parse the result
        return self._parse_result(result)
    
    def _parse_result(self, crew_result) -> HookResult:
        """Parse the crew result into a HookResult object."""
        try:
            # Get the raw output
            raw_output = str(crew_result)
            
            # Try to extract JSON from the output
            # Handle case where output might have markdown or extra text
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = raw_output[json_start:json_end]
                result_data = json.loads(json_str)
                
                winner = result_data.get('winner', {})
                
                # Calculate scores from hooks_evaluated
                scores = {}
                for hook_eval in result_data.get('hooks_evaluated', []):
                    if hook_eval.get('number') == winner.get('number'):
                        scores = {
                            'emotional_charge': hook_eval.get('emotional', 0),
                            'controversy_potential': hook_eval.get('controversy', 0),
                            'negativity_bias': hook_eval.get('negativity', 0),
                            'defensibility': hook_eval.get('defensibility', 0)
                        }
                        break
                
                return HookResult(
                    hook_text=winner.get('hook', 'Could not generate hook'),
                    subject_word=winner.get('subject_word'),
                    object_word=winner.get('object_word'),
                    scores=scores,
                    reasoning=winner.get('reasoning', '')
                )
            else:
                # Fallback: just use the raw output as hook text
                print("[WARNING] Could not parse JSON from crew result")
                return HookResult(
                    hook_text=raw_output[:100],  # First 100 chars
                    subject_word=None,
                    object_word=None,
                    scores={},
                    reasoning="Parsing failed"
                )
                
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON parse error: {e}")
            return HookResult(
                hook_text="Failed to generate hook",
                subject_word=None,
                object_word=None,
                scores={},
                reasoning=str(e)
            )
        except Exception as e:
            print(f"[WARNING] Error parsing result: {e}")
            return HookResult(
                hook_text="Failed to generate hook",
                subject_word=None,
                object_word=None,
                scores={},
                reasoning=str(e)
            )


class HookRenderer:
    """
    Renders hook text onto video frames with SUBJECT (yellow) and OBJECT (purple) highlights.
    """
    
    def __init__(self, font_path: str = "BebasNeue-Regular.ttf", font_size: int = 42):
        self.font_path = font_path
        self.font_size = font_size
        
        # Colors
        self.default_color = (255, 255, 255)  # White
        self.subject_color = (255, 255, 0)     # Yellow for SUBJECT
        self.object_color = (180, 100, 255)    # Purple for OBJECT
        
        # Import PIL for text rendering
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.Image = Image
            self.ImageDraw = ImageDraw
            self.ImageFont = ImageFont
        except ImportError:
            raise ImportError("Please install Pillow: pip install Pillow")
    
    def create_hook_image(self, hook_result: HookResult, max_width: int = 560) -> Any:
        """
        Create a hook text image with highlighted SUBJECT and OBJECT.
        
        Args:
            hook_result: The hook result with text and highlights
            max_width: Maximum width for the text
            
        Returns:
            BGRA numpy array image
        """
        import numpy as np
        
        # Load font
        try:
            font = self.ImageFont.truetype(self.font_path, self.font_size)
        except:
            font = self.ImageFont.load_default()
        
        hook_text = hook_result.hook_text.upper()  # Uppercase for impact
        subject = hook_result.subject_word.upper() if hook_result.subject_word else None
        obj = hook_result.object_word.upper() if hook_result.object_word else None
        
        # Split text into words with colors
        words = hook_text.split()
        word_colors = []
        
        for word in words:
            clean_word = word.strip('.,!?;:\'"')
            if subject and clean_word == subject:
                word_colors.append((word, self.subject_color))
            elif obj and clean_word == obj:
                word_colors.append((word, self.object_color))
            else:
                word_colors.append((word, self.default_color))
        
        # Calculate sizes
        temp_img = self.Image.new('RGBA', (1, 1))
        temp_draw = self.ImageDraw.Draw(temp_img)
        
        # Calculate word widths and total width
        word_widths = []
        space_width = temp_draw.textlength(" ", font=font)
        
        for word, _ in word_colors:
            width = temp_draw.textlength(word, font=font)
            word_widths.append(width)
        
        # Line wrapping
        lines = []
        current_line = []
        current_width = 0
        
        for i, (word, color) in enumerate(word_colors):
            word_width = word_widths[i]
            
            if current_width + word_width + (space_width if current_line else 0) > max_width and current_line:
                lines.append(current_line)
                current_line = [(word, color, word_width)]
                current_width = word_width
            else:
                current_line.append((word, color, word_width))
                current_width += word_width + (space_width if len(current_line) > 1 else 0)
        
        if current_line:
            lines.append(current_line)
        
        # Calculate total height
        bbox = temp_draw.textbbox((0, 0), "Hg", font=font)
        line_height = bbox[3] - bbox[1] + 8
        total_height = line_height * len(lines) + 20
        total_width = max_width + 40
        
        # Create actual image
        img = self.Image.new('RGBA', (total_width, total_height), (0, 0, 0, 0))
        draw = self.ImageDraw.Draw(img)
        
        # Draw each line
        y = 10
        for line in lines:
            # Calculate line width for centering
            line_total_width = sum(w for _, _, w in line) + space_width * (len(line) - 1)
            x = (total_width - line_total_width) // 2
            
            for word, color, width in line:
                # Draw shadow
                draw.text((x + 2, y + 2), word, font=font, fill=(0, 0, 0, 200))
                # Draw text
                draw.text((x, y), word, font=font, fill=(*color, 255))
                x += width + space_width
            
            y += line_height
        
        # Convert to numpy array (BGRA for OpenCV)
        img_array = np.array(img)
        # Convert RGBA to BGRA
        img_bgra = img_array[:, :, [2, 1, 0, 3]]
        
        return img_bgra
    
    def overlay_hook_on_frame(self, frame: Any, hook_image: Any, 
                               y_position: int = 30) -> Any:
        """
        Overlay hook text image on a video frame.
        
        Args:
            frame: Video frame (BGR)
            hook_image: Hook text image (BGRA)
            y_position: Fixed pixel position from top (default 30px)
            
        Returns:
            Frame with hook overlay
        """
        import numpy as np
        
        frame_height, frame_width = frame.shape[:2]
        hook_height, hook_width = hook_image.shape[:2]
        
        # Center horizontally on frame
        x = (frame_width - hook_width) // 2
        # Fixed position from top (within gradient area)
        y = y_position
        
        # Clamp positions to valid range
        x = max(0, x)
        y = max(0, min(y, frame_height - hook_height))
        
        # Calculate overlay region
        x_end = min(x + hook_width, frame_width)
        y_end = min(y + hook_height, frame_height)
        
        # Adjust hook image if needed
        hook_crop_w = x_end - x
        hook_crop_h = y_end - y
        hook_cropped = hook_image[:hook_crop_h, :hook_crop_w]
        
        if hook_cropped.shape[0] == 0 or hook_cropped.shape[1] == 0:
            return frame
        
        # Simple alpha blending - NO black background
        alpha = hook_cropped[:, :, 3] / 255.0
        alpha = np.stack([alpha] * 3, axis=-1)
        
        hook_rgb = hook_cropped[:, :, :3]
        roi = frame[y:y_end, x:x_end]
        
        blended = (alpha * hook_rgb + (1 - alpha) * roi).astype(np.uint8)
        frame[y:y_end, x:x_end] = blended
        
        return frame


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    # Test the hook generator
    test_transcript = """
    You know what's crazy? People spend 40 years at jobs they hate, 
    saving for a retirement they might not live to see. And then they call 
    entrepreneurs "risky". The real risk is trading your entire life for a 
    pension that might get cut. Nobody talks about that.
    """
    
    print("Testing Hook Generator...")
    print("=" * 60)
    
    try:
        generator = HookGeneratorCrew()
        result = generator.generate_hook(test_transcript)
        
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(f"Hook: {result.hook_text}")
        print(f"Subject (Yellow): {result.subject_word}")
        print(f"Object (Purple): {result.object_word}")
        print(f"Scores: {result.scores}")
        print(f"Reasoning: {result.reasoning}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
