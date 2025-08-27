
### docs/user-guide/run-with-docker.md
```markdown
---
title: Run with Docker
parent: User Guide
nav_order: 3
---
```
# Run with Docker

## Pull & run
```bash
docker pull yourname/giftbench:v1.2.1
docker run --name giftbench --gpus all -p 7860:7860 yourname/giftbench:v1.2.1
```

