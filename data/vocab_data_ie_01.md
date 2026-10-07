# Vocabulary data - Ireland

> [!NOTE]
> Revised with assistance from OpenAI Codex/GPT-6.

This data was collected by Down Syndrome Ireland as part of a project from 2019 to 2020.

## Measures and preparation

The source records DSE checklist totals and counts for Checklists 1, 2 and 3 at the start and end of the project. Comprehension, imitation and speech are separate measures. The models use the says totals for speech. They do not add imitation counts to those totals.

The baseline wave omitted Checklist 3. Its zeros mark items that were not assessed, rather than showing that a child knew none of those words. The loader marks that wave with a 460-word form ceiling, and the primary analysis masks its outcomes under `INCOMPLETE_ADMINISTRATION_CEILINGS` in `vocab_growth.data_utils`. Follow-up records can contribute subject to the other data rules. This differs from the retained partial form in [ie_02](vocab_data_ie_02.md), whose asymmetrical treatment is documented there.

A comprehension count below the greater of recorded production and speech is masked under the shared nesting rule. Speech is retained. The [data guide](readme.md) distinguishes prepared source rows from rows used in a particular model.

## Fields

- subject_id
- age_months_start
- age_months_end
- hearing_status
- fluctuating_consistent
- hearing_aid
- vision_status
- glasses
- see_learn_experience
- understands_1_start
- understands_2_start
- understands_3_start
- understands_total_start
- understands_1_end
- understands_2_end
- understands_3_end
- understands_total_end
- imitates_1_start
- imitates_2_start
- imitates_3_start
- imitates_total_start
- imitates_1_end
- imitates_2_end
- imitates_3_end
- imitates_total_end
- says_1_start
- says_2_start
- says_3_start
- says_total_start
- says_1_end
- says_2_end
- says_3_end
- says_total_end

## License

This data is licensed under the Creative Commons Attribution 4.0 International (CC BY 4.0) See [LICENSE](LICENSE) for details.