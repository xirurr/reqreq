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

    read_files = []
    for file in files:
        content = await file.read()
        read_files.append({"filename": file.filename, "content": content})

    # Сортируем по имени файла
    read_files.sort(key=lambda x: x['filename'])

    analyzer = UniversalMultimodalAnalyzer(config, release_version, read_files)

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
        "files_processed": [file['filename'] for file in read_files],
        "summary": {
            "entities_found": len(results.get("entities", [])),
            "requirements_found": len(results.get("requirements", [])),
            "dependencies_found": len(results.get("dependencies", []))
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)