import os
import tempfile
import traceback
from typing import List

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form

from configs.llm_configs import lm_studio_config
from configs.neo4j_config import services_config
from universal_document_analyzer import UniversalMultimodalAnalyzer

app = FastAPI()

@app.post("/analyze_release/")
async def analyze_release(release_version: str = Form(...), files: List[UploadFile] = File(...)):
    if not files:
        return {"error": "No files uploaded"}

    # Корректно объединяем конфигурации
    config = {**lm_studio_config, **services_config}

    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for file in files:
            file_path = os.path.join(temp_dir, file.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await file.read())
            file_paths.append(file_path)

        file_paths.sort()

        analyzer = UniversalMultimodalAnalyzer(config, release_version, file_paths)

        print(f"Начало анализа для релиза: {release_version}")
        try:
            results = analyzer.analyze()
        except Exception as e:
            print("Сломался")
            print(traceback.format_exc())  # выводит полный traceback
            raise
        print("Анализ завершен.")

    return {
        "release_version": release_version,
        "files_processed": [file.filename for file in files],
        "summary": {
            "entities_found": len(results.get("entities", [])),
            "requirements_found": len(results.get("requirements", [])),
            "dependencies_found": len(results.get("dependencies", []))
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)