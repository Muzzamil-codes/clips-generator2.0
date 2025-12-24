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
    subject_words: Optional[str]  # Word(s) to highlight in YELLOW (can be multiple consecutive words)
    object_words: Optional[str]   # Word(s) to highlight in PURPLE (can be multiple consecutive words)
    scores: Dict[str, float]      # Evaluation scores
    reasoning: str                # Why this hook was selected
    prefix: str = ""              # Prefix to add before the hook (e.g., "$8B Fintech CEO: ")


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
            goal="Create 8 story-telling hooks that build curiosity and make viewers NEED to watch the full video",
            backstory="""You are a master storyteller who creates hooks that feel like the opening line of an incredible story.
Your hooks make viewers feel like they're about to discover a secret, witness a transformation, or learn something that will change their perspective.

You write hooks as if the speaker is about to share something personal and powerful:
- REVELATION: "I discovered something that changed everything..."
- TRANSFORMATION: "This one shift took me from X to Y..."
- INSIDER SECRET: "Nobody talks about this, but..."
- CHALLENGE: "Everyone told me I was wrong, until..."
- CURIOSITY GAP: "The real reason why X happens is not what you think..."
- STORY OPENER: "When I first heard this, I didn't believe it..."

Your hooks:
- Sound like the START of an interesting story, not a comment ABOUT the video
- Make the viewer feel like they'll miss out if they scroll past
- Create an open loop that MUST be closed by watching
- Feel personal and authentic, like a friend sharing a secret
- Are NEVER sarcastic or dismissive - they're inviting and intriguing

Your hooks are always under 12 words and designed to maximize watch time by creating irresistible curiosity.""",
            llm="gemini/gemini-2.5-pro",
            verbose=True,
            allow_delegation=False,
            temperature=0.7
        )
    
    def _create_evaluator_agent(self):
        """Create the hook evaluator agent using Gemini 2.0 Flash."""
        return self.Agent(
            role="Engagement Analyst",
            goal="Evaluate hooks and identify the most curiosity-inducing option with SUBJECT/OBJECT highlights",
            backstory="""You are an expert in viewer psychology and content engagement.
You analyze hooks based on their potential to:
1. Create irresistible curiosity (viewers MUST know what happens next)
2. Feel like the start of a valuable story
3. Promise a transformation, secret, or insight
4. Make viewers feel they'll miss out if they scroll

You identify hooks that sound like authentic storytelling, not commentary.
The best hooks make viewers think "I need to hear this" not "what are they talking about?"

You also identify the KEY SUBJECT (main topic - highlighted YELLOW) and 
KEY OBJECT (what's being said about it - highlighted PURPLE) in each hook
to maximize visual impact and comprehension.""",
            llm="gemini/gemini-3-pro-preview",
            verbose=True,
            allow_delegation=False,
            temperature=0
        )
    
    def _create_generation_task(self, agent, transcript: str):
        """Create the hook generation task."""
        return self.Task(
            description=f"""Based on this podcast transcript, create exactly 8 hook variations.
Each hook MUST be 12 words or fewer.

TRANSCRIPT:
{transcript}

Create hooks using these STORYTELLING styles (at least one of each, you can repeat if needed):
1. REVELATION - Share a discovery: "I found out why most people fail at..."
2. TRANSFORMATION - Show change: "This one thing changed how I see..."  
3. INSIDER SECRET - Exclusive knowledge: "Nobody tells you this about..."
4. CHALLENGE - Overcome opposition: "Everyone said I was crazy until..."
5. CURIOSITY GAP - Create intrigue: "The real reason behind X isn't what you think..."
6. PERSONAL STORY - Relatable opening: "When I first learned this, I couldn't believe..."
7. BOLD CLAIM - Confident statement: "This is the fastest way to..."
8. QUESTION HOOK - Engage directly: "What if everything you knew about X was wrong?"

RULES:
- Each hook MUST be under 12 words
- Hooks should sound like the BEGINNING of a story, making viewers want to hear more
- They should feel like the speaker is about to share something valuable/personal
- NEVER sound like a comment on the video - sound like you're IN the video telling a story
- Make viewers feel they'll miss out if they don't watch
- Number each hook 1-8

Output format:
1. [STYLE]: "Hook text here"
2. [STYLE]: "Hook text here"
... etc for all 8 hooks""",
            expected_output="""A numbered list of exactly 8 hooks, each labeled with its style type,
each under 12 words, designed to create curiosity and maximize watch time.""",
            agent=agent
        )
    
    def _create_evaluation_task(self, agent, generation_task):
        """Create the hook evaluation task."""
        return self.Task(
            description="""Evaluate all 8 hooks from the previous task.

Score each hook on a scale of 1-10 for:
1. CURIOSITY: How strongly does this make viewers want to know more? (open loop strength)
2. STORY_FEEL: Does this feel like the start of a compelling story being told?
3. VALUE_PROMISE: Does it promise insight, transformation, or exclusive knowledge?
4. AUTHENTICITY: Does it sound natural and personal, not like marketing/clickbait?

Then select the BEST hook (highest combined score).

For the winning hook, identify:
- SUBJECT_WORDS: The main topic - can be ONE OR MORE CONSECUTIVE WORDS (to highlight in YELLOW)
  Example: for "I discovered why successful people wake up early", subject could be "successful people" (2 words together)
- OBJECT_WORDS: The key action/descriptor - can be ONE OR MORE CONSECUTIVE WORDS (to highlight in PURPLE)
  Example: for "I discovered why successful people wake up early", object could be "wake up early" (3 words together)

IMPORTANT: Choose the most impactful CONSECUTIVE words to highlight together. Do NOT skip words in between.

OUTPUT YOUR RESPONSE AS VALID JSON with this exact structure:
{
    "hooks_evaluated": [
        {"number": 1, "hook": "hook text", "curiosity": 8, "story_feel": 7, "value_promise": 6, "authenticity": 9, "total": 30},
        ... for all 8 hooks
    ],
    "winner": {
        "number": 1,
        "hook": "the winning hook text",
        "subject_words": "TOPIC WORDS",
        "object_words": "DESCRIPTOR WORDS",
        "reasoning": "Why this hook will make viewers want to watch the full video"
    }
}

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no extra text.""",
            expected_output="""A JSON object containing evaluation scores for all 8 hooks
and the winning hook with SUBJECT and OBJECT words identified.""",
            agent=agent,
            context=[generation_task]
        )
    
    def generate_hook(self, transcript: str, use_discord: bool = True, discord_timeout: int = 1200) -> HookResult:
        """
        Generate the best hook for a given transcript.
        
        Flow:
        1. Generate 8 hook variations using AI
        2. If use_discord=True, post hooks to Discord and wait for user selection
        3. If user selects a hook or provides custom one, use that
        4. If timeout or use_discord=False, use AI evaluator to select best hook
        
        Args:
            transcript: The podcast transcript text
            use_discord: Whether to use Discord for hook selection (default True)
            discord_timeout: Seconds to wait for Discord response (default 1200 = 20 min)
            
        Returns:
            HookResult with the winning hook and highlights
        """
        print("[HOOK] Initializing CrewAI agents...")
        
        # Create generator agent
        generator = self._create_generator_agent()
        
        # Create generation task
        generation_task = self._create_generation_task(generator, transcript)
        
        # Create crew for JUST generation (not evaluation yet)
        generation_crew = self.Crew(
            agents=[generator],
            tasks=[generation_task],
            process=self.Process.sequential,
            verbose=True
        )
        
        print("[HOOK] Running hook generation...")
        
        # Execute generation
        generation_result = generation_crew.kickoff()
        raw_hooks_output = str(generation_result)
        
        print(f"[HOOK] Generated hooks:\n{raw_hooks_output}")
        
        # Parse the generated hooks
        from hook_discord_selector import parse_hooks_from_generator_output, run_hook_selector_bot
        hooks = parse_hooks_from_generator_output(raw_hooks_output)
        
        if not hooks:
            print("[HOOK] Failed to parse hooks, falling back to AI evaluation")
            use_discord = False
        else:
            print(f"[HOOK] Parsed {len(hooks)} hooks successfully")
        
        # Try Discord selection if enabled
        selected_hook_text = None
        if use_discord and hooks:
            try:
                print(f"[HOOK] Posting hooks to Discord (timeout: {discord_timeout}s)...")
                discord_result = run_hook_selector_bot(hooks, timeout_seconds=discord_timeout, transcript=transcript)
                
                if not discord_result.timed_out and discord_result.selected_hook:
                    selected_hook_text = discord_result.selected_hook
                    print(f"[HOOK] User selected hook: {selected_hook_text}")
                    
                    if discord_result.is_custom:
                        print("[HOOK] Using custom hook from user")
                        # For custom hooks, we need to identify subject/object words
                        # Use the evaluator for just this task
                        return self._evaluate_single_hook(selected_hook_text)
                    else:
                        # User selected a numbered hook, now evaluate just that one
                        print(f"[HOOK] User selected hook #{discord_result.hook_number}")
                        return self._evaluate_single_hook(selected_hook_text)
                else:
                    print("[HOOK] Discord timed out, falling back to AI evaluation")
            except Exception as e:
                print(f"[HOOK] Discord error: {e}, falling back to AI evaluation")
        
        # Fall back to AI evaluation
        print("[HOOK] Using AI evaluator to select best hook...")
        
        # Create evaluator agent
        evaluator = self._create_evaluator_agent()
        
        # Create evaluation task with the generated hooks
        evaluation_task = self._create_evaluation_task_from_hooks(evaluator, raw_hooks_output)
        
        # Create crew for evaluation
        evaluation_crew = self.Crew(
            agents=[evaluator],
            tasks=[evaluation_task],
            process=self.Process.sequential,
            verbose=True
        )
        
        print("[HOOK] Running hook evaluation...")
        result = evaluation_crew.kickoff()
        
        # Parse the result
        return self._parse_result(result)
    
    def _evaluate_single_hook(self, hook_text: str) -> HookResult:
        """
        Evaluate a single hook to identify subject/object words.
        
        Args:
            hook_text: The hook text to evaluate
            
        Returns:
            HookResult with subject/object words identified
        """
        evaluator = self._create_evaluator_agent()
        
        task = self.Task(
            description=f"""For this hook, identify the key words to highlight:

HOOK: "{hook_text}"

Identify:
- SUBJECT_WORDS: The main topic - can be ONE OR MORE CONSECUTIVE WORDS (to highlight in YELLOW)
- OBJECT_WORDS: The key descriptor/action - can be ONE OR MORE CONSECUTIVE WORDS (to highlight in PURPLE)

OUTPUT YOUR RESPONSE AS VALID JSON with this exact structure:
{{
    "hook": "{hook_text}",
    "subject_words": "TOPIC WORDS",
    "object_words": "DESCRIPTOR WORDS",
    "reasoning": "Why these words were chosen for highlighting"
}}

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no extra text.""",
            expected_output="A JSON object with the hook and highlighted words identified.",
            agent=evaluator
        )
        
        crew = self.Crew(
            agents=[evaluator],
            tasks=[task],
            process=self.Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        raw_output = str(result)
        
        try:
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = raw_output[json_start:json_end]
                data = json.loads(json_str)
                return HookResult(
                    hook_text=data.get('hook', hook_text),
                    subject_words=data.get('subject_words'),
                    object_words=data.get('object_words'),
                    scores={},
                    reasoning=data.get('reasoning', ''),
                    prefix=""
                )
        except Exception as e:
            print(f"[HOOK] Error parsing single hook evaluation: {e}")
        
        # Fallback
        return HookResult(
            hook_text=hook_text,
            subject_words=None,
            object_words=None,
            scores={},
            reasoning="Manual selection",
            prefix=""
        )
    
    def _create_evaluation_task_from_hooks(self, agent, hooks_text: str):
        """Create evaluation task from already-generated hooks text."""
        return self.Task(
            description=f"""Evaluate these hooks that were already generated:

{hooks_text}

Score each hook on a scale of 1-10 for:
1. EMOTIONAL_CHARGE: How strongly will this trigger an emotional response?
2. CONTROVERSY_POTENTIAL: How likely to generate debate/arguments?
3. NEGATIVITY_BIAS: How effectively does it use negativity to drive engagement?
4. DEFENSIBILITY: Can you defend posting this? (10 = perfectly defensible, 1 = likely to get banned)

Then select the BEST hook (highest combined score with defensibility >= 7).

For the winning hook, identify:
- SUBJECT_WORDS: The main topic - can be ONE OR MORE CONSECUTIVE WORDS (to highlight in YELLOW)
- OBJECT_WORDS: The key descriptor/action - can be ONE OR MORE CONSECUTIVE WORDS (to highlight in PURPLE)

OUTPUT YOUR RESPONSE AS VALID JSON with this exact structure:
{{
    "hooks_evaluated": [
        {{"number": 1, "hook": "hook text", "emotional": 8, "controversy": 7, "negativity": 6, "defensibility": 9, "total": 30}},
        ... for all 8 hooks
    ],
    "winner": {{
        "number": 1,
        "hook": "the winning hook text",
        "subject_words": "TOPIC WORDS",
        "object_words": "DESCRIPTOR WORDS",
        "reasoning": "Why this hook will perform best"
    }}
}}

IMPORTANT: Return ONLY the JSON object, no markdown formatting, no extra text.""",
            expected_output="""A JSON object containing evaluation scores for all hooks
and the winning hook with SUBJECT and OBJECT words identified.""",
            agent=agent
        )
    
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
                    subject_words=winner.get('subject_words') or winner.get('subject_word'),  # Backward compat
                    object_words=winner.get('object_words') or winner.get('object_word'),    # Backward compat
                    scores=scores,
                    reasoning=winner.get('reasoning', ''),
                    prefix=""  # Default empty, can be set by caller
                )
            else:
                # Fallback: just use the raw output as hook text
                print("[WARNING] Could not parse JSON from crew result")
                return HookResult(
                    hook_text=raw_output[:100],  # First 100 chars
                    subject_words=None,
                    object_words=None,
                    scores={},
                    reasoning="Parsing failed",
                    prefix=""
                )
                
        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON parse error: {e}")
            return HookResult(
                hook_text="Failed to generate hook",
                subject_words=None,
                object_words=None,
                scores={},
                reasoning=str(e),
                prefix=""
            )
        except Exception as e:
            print(f"[WARNING] Error parsing result: {e}")
            return HookResult(
                hook_text="Failed to generate hook",
                subject_words=None,
                object_words=None,
                scores={},
                reasoning=str(e),
                prefix=""
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
        
        # Add prefix if present
        full_hook_text = ""
        if hook_result.prefix:
            full_hook_text = hook_result.prefix + hook_result.hook_text
        else:
            full_hook_text = hook_result.hook_text
        
        hook_text = full_hook_text.upper()  # Uppercase for impact
        
        # Handle consecutive word highlighting
        subject_words = hook_result.subject_words.upper().split() if hook_result.subject_words else []
        object_words = hook_result.object_words.upper().split() if hook_result.object_words else []
        
        # Split text into words with colors
        words = hook_text.split()
        word_colors = []
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = word.strip('.,!?;:\'"')
            matched = False
            
            # Check for subject phrase match (consecutive words)
            if subject_words and i + len(subject_words) <= len(words):
                phrase_match = True
                for j, subj_word in enumerate(subject_words):
                    check_word = words[i + j].strip('.,!?;:\'"')
                    if check_word != subj_word:
                        phrase_match = False
                        break
                if phrase_match:
                    # Add all words in the phrase with subject color
                    for j in range(len(subject_words)):
                        word_colors.append((words[i + j], self.subject_color))
                    i += len(subject_words)
                    matched = True
            
            # Check for object phrase match (consecutive words)
            if not matched and object_words and i + len(object_words) <= len(words):
                phrase_match = True
                for j, obj_word in enumerate(object_words):
                    check_word = words[i + j].strip('.,!?;:\'"')
                    if check_word != obj_word:
                        phrase_match = False
                        break
                if phrase_match:
                    # Add all words in the phrase with object color
                    for j in range(len(object_words)):
                        word_colors.append((words[i + j], self.object_color))
                    i += len(object_words)
                    matched = True
            
            if not matched:
                word_colors.append((word, self.default_color))
                i += 1
        
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
        
        # Test with prefix
        result.prefix = "$8B Fintech CEO: "
        
        print("\n" + "=" * 60)
        print("RESULT:")
        print("=" * 60)
        print(f"Hook: {result.hook_text}")
        print(f"Prefix: {result.prefix}")
        print(f"Full Hook: {result.prefix + result.hook_text}")
        print(f"Subject (Yellow): {result.subject_words}")
        print(f"Object (Purple): {result.object_words}")
        print(f"Scores: {result.scores}")
        print(f"Reasoning: {result.reasoning}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
