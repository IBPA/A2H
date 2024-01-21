Pipeline:

- Query relevant articles from PubMed: https://pubmed.ncbi.nlm.nih.gov/

- Save result abstracts

- Run `pubmed2abstract.py` and provide the path to abstract result file as the first argument.  This will create a parsed_files directory with one file per abstract.

- (if necessary) Query the list of interventions from this PostgreSQL database using `interventions.sql` and save the results to `interventions_raw.csv`
    - aact-db.ctti-clinicaltrials.org

- (if necessary) Resolve the intervention names in `interventions_raw.csv` to their ingredient and brand names using `getRelatedInterventionNames.py`.

- Filter the pubmed abstracts using `filterAbstracts.py` and `interventions_processed.csv`.

- The files in `filtered_files` should then be ready for annotation.