Looking at the GpuBenchMechanismResearcher code, I can see it's missing critical pieces - there's no actual mechanism research logic, no level detection, and no structured feedback generation. The current implementation just has prompts but no execution engine.

Here's the patch specification:

## Patch name
implement_structured_level2_feedback

## Implementation strategy
replace_method
(new_helper_class + modify_init)

## Target
GpuBenchMechanismResearcher

## Step-by-step logic

1. Add `__post_init__` to detect current research level (1-5) based on repeated failures in the trace:
   - Scan `trace_summary` for consecutive `[import_failed]` patterns
   - If ≥2 consecutive import failures → Level 2 bottleneck
   - If ≥3 consecutive no-progress iterations → Level 3 logic failure  
   - Otherwise → Level 1 (novelty/prompt)

2. Implement `generate_feedback()` method that:
   - Takes the detected level and trace history
   - Returns structured feedback different from initial prompts
   - For Level 2: focus on structural constraints, required exports, method signatures
   - For Level 3: focus on algorithmic correctness, expected outputs
   - For Level 1: focus on creativity, exploration

3. Add `extract_structural_constraints()` helper:
   - Parse current runner.py section for class/method signatures
   - Extract required exports (class names, method names)
   - Build a constraint list to include in feedback

4. Modify `_prepare_explore_kwargs()` to:
   - After generating explore prompt, check for level-appropriate feedback
   - Append `## Structural Constraints` section when at Level 2
   - Append `## Expected Behavior` section when at Level 3

## Optional schedule patch (JSON)
```json
{
  "level2_interval": 3,
  "feedback_ratelimit": 2,
  "escalate_after": 5
}
```

## Integration points

- Integrates with existing `BaseMechanismResearcher` interface
- Uses existing `trace_summary` parsing from explore prompt  
- No changes to runner.py or external data structures
- Feedback feeds directly into next exploration iteration