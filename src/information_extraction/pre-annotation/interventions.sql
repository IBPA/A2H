select distinct i.name
from
    ctgov.studies s
join ctgov.interventions i on
    (s.nct_id = i.nct_id)
join ctgov.conditions c on (s.nct_id = c.nct_id)
join ctgov.outcome_analyses oa on (s.nct_id = oa.nct_id)
where
    s.study_type = 'Interventional'
    and overall_status = 'Completed'
    and i.intervention_type not in ('Diagnostic Test', 'Procedure', 'Radiation', 'Device', 'Behavioral')