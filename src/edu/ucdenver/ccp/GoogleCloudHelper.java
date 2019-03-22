package edu.ucdenver.ccp;

import com.google.cloud.bigquery.*;

import java.util.UUID;

public class GoogleCloudHelper {

	public static Iterable<FieldValueList> runQuery(String query) throws InterruptedException {
		BigQuery bigquery = BigQueryOptions.getDefaultInstance().getService();
		QueryJobConfiguration queryConfig =
				QueryJobConfiguration.newBuilder(query)
						// Use standard SQL syntax for queries.
						// See: https://cloud.google.com/bigquery/sql-reference/
						.setUseLegacySql(false)
						.build();

		// Create a job ID so that we can safely retry.
		JobId jobId = JobId.of(UUID.randomUUID().toString());
		Job queryJob = bigquery.create(JobInfo.newBuilder(queryConfig).setJobId(jobId).build());

		// Wait for the query to complete.
		queryJob = queryJob.waitFor();

		// Check for errors
		if (queryJob == null) {
			throw new RuntimeException("Job no longer exists");
		} else if (queryJob.getStatus().getError() != null) {
			// You can also look at queryJob.getStatus().getExecutionErrors() for all
			// errors, not just the latest one.
			throw new RuntimeException(queryJob.getStatus().getError().toString());
		}

		// Get the results.
		QueryResponse response = bigquery.getQueryResults(jobId);

		TableResult result = queryJob.getQueryResults();

		return result.iterateAll();

		// Print all pages of the results.
//		for (FieldValueList row : result.iterateAll()) {
//			String url = row.get("url").getStringValue();
//			long viewCount = row.get("view_count").getLongValue();
//			System.out.printf("url: %s views: %d%n", url, viewCount);
//		}
	}

}
