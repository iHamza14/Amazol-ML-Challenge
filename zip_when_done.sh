#!/bin/bash
echo "Waiting for pipeline to finish..."
# Wait for task-43 to finish or just assume it finishes when python is no longer running pipeline.py
while pgrep -f "pipeline.py" > /dev/null; do
    sleep 60
done
echo "Pipeline finished. Creating submission zip..."
cd student_resource
zip -r ../Antigravity_submission.zip output code Documentation_template.md
echo "Zip created at Antigravity_submission.zip"
