from typing import Tuple, List

from services.CacheService import CacheService


class DocumentProcessor:
    def __init__(self, temp_dir: str, cache: CacheService):
        self.temp_dir = temp_dir
        self.cache = cache

    def process_docx(self, file_path: str) -> Tuple[str, List[dict]]:
        """
        Извлекает текст и информацию о всех изображениях из DOCX файла
        через XML структуру документа
        """
        import zipfile
        from PIL import Image
        import os
        from xml.etree import ElementTree as ET

        images_info = []
        output_dir = "parsed/images"
        os.makedirs(output_dir, exist_ok=True)

        try:
            with zipfile.ZipFile(file_path) as docx_zip:
                filename = os.path.basename(file_path)
                filename_without_extension = os.path.splitext(os.path.basename(file_path))[0]

                # Читаем document.xml
                doc_xml = docx_zip.read('word/document.xml')

                doc_hash = "all_doc:"+self.cache.generate_hash(doc_xml)
                # if self.cache.exists(doc_hash):
                #     return self.cache.get(doc_hash)

                tree = ET.fromstring(doc_xml)

                namespaces = {
                    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
                    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
                    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
                    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture'
                }

                # Находим все изображения в XML
                for idx, drawing in enumerate(tree.findall('.//w:drawing', namespaces)):
                    image_info = {
                        'index': idx + 1
                    }

                    # Получаем ID изображения
                    pic_element = drawing.find('.//pic:blipFill/a:blip', namespaces)
                    if pic_element is not None:
                        embed_id = pic_element.get(
                            '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                        if embed_id:
                            image_info['embed_id'] = embed_id

                    # Получаем метаданные изображения
                    docPr = drawing.find('.//wp:docPr', namespaces)
                    if docPr is not None:
                        image_info['title'] = docPr.get('title', '')
                        image_info['description'] = docPr.get('descr', '')

                    images_info.append(image_info)

                # Читаем связи
                rels_xml = docx_zip.read('word/_rels/document.xml.rels')
                rels_tree = ET.fromstring(rels_xml)
                rel_namespace = {'rel': 'http://schemas.openxmlformats.org/package/2006/relationships'}

                # Мапим ID на реальные файлы
                rels = {
                    rel.get('Id'): rel.get('Target')
                    for rel in rels_tree.findall('.//rel:Relationship', rel_namespace)
                    if rel.get('Target', '').startswith('media/')
                }

                # Добавляем информацию о файлах
                for img_info in images_info:
                    if 'embed_id' in img_info and img_info['embed_id'] in rels:
                        img_path = f"word/{rels[img_info['embed_id']]}"
                        img_info['original_path'] = img_path

                        # Для сохранения используем только имя файла без 'media/'
                        save_name = os.path.basename(rels[img_info['embed_id']])
                        save_dir = os.path.join(output_dir, filename_without_extension)
                        save_path = os.path.join(output_dir, filename_without_extension, save_name)
                        os.makedirs(save_dir, exist_ok=True)

                        # Извлекаем файл
                        with docx_zip.open(img_path) as source:
                            with open(save_path, 'wb') as target:
                                target.write(source.read())

                        full_path = os.path.join(output_dir, img_path)
                        img_info['saved_path'] = save_path

                        # Конвертируем в PNG если нужно
                        if not full_path.lower().endswith('.png'):
                            try:
                                img = Image.open(full_path)
                                new_path = os.path.splitext(save_path)[0] + '.png'
                                img.save(new_path, 'PNG')
                                os.remove(full_path)
                                img_info['saved_path'] = new_path
                            except Exception as e:
                                print(f"Ошибка конвертации изображения: {str(e)}")

                # Получаем текст
                doc_text = "\n".join(self._get_paragraph_content(p, namespaces, images_info)
                                     for p in tree.findall('.//w:p', namespaces))

                self.cache.set(doc_hash, (doc_text, images_info))
                return doc_text, images_info

        except Exception as e:
            print(f"Ошибка при обработке DOCX файла: {str(e)}")
            return "", []

    def _get_paragraph_content(self, paragraph, namespaces, images_info):
        """Извлекает текст и ссылки на изображения из параграфа"""
        content = []

        for child in paragraph.iter():
            # Если это текстовый элемент
            if child.tag == f"{{{namespaces['w']}}}t" and child.text:
                content.append(child.text)

            # Если это drawing (изображение)
            elif child.tag == f"{{{namespaces['w']}}}drawing":
                # Находим ID изображения
                pic_element = child.find('.//pic:blipFill/a:blip', namespaces)
                if pic_element is not None:
                    embed_id = pic_element.get(
                        '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
                    if embed_id:
                        # Ищем информацию об изображении
                        for img in images_info:
                            if img.get('embed_id') == embed_id and 'saved_path' in img:
                                # Добавляем маркер изображения в текст
                                img_marker = f"\n[IMAGE:{img['saved_path']}]"
                                if 'description' in img and img['description']:
                                    img_marker += f" - {img['description']}"
                                content.append(img_marker)
                                break

        return ''.join(content)