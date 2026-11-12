def markdown_report(report, snapshot):
    lines = [
        "# Code review", "", f"Source: {report.source}", f"Revision: {report.revision}",
        f"Review status: {'completed within available scope' if report.complete else 'INCOMPLETE'}",
        "Reviewers: " + ", ".join(result.agent for result in report.agents), "",
    ]
    for finding in report.findings:
        url = snapshot.source_url(finding)
        location = f"{finding.path}:{finding.line}"
        lines.extend([
            f"## {finding.severity.value}: {finding.title}", "",
            f"Location: [{location}]({url})" if url else f"Location: {location}",
            f"Reviewer: {finding.agent} | Confidence: {finding.confidence:.0%}", "",
            "Evidence:", "```", finding.evidence, "```", "",
            f"Impact: {finding.impact}", "", f"Proposed fix: {finding.suggested_fix}", "",
            "Suggested tests:",
            *[f"- {test}" for test in finding.suggested_tests], "",
        ])
    lines.extend(["## Limitations", "", *[f"- {s}" for s in report.limitations]])
    return "\n".join(lines) + "\n"
