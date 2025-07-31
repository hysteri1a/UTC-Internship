from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.responses import StreamingResponse
from typing import  Dict, Any, List
from uuid import uuid4
import shutil
import os
from pathlib import Path
from datetime import datetime
from docx import Document
from read import ReviewSystem
import json
from typing import AsyncGenerator
from function import validate_session_exists, validate_file_type, sessions,get_upload_params, ALL_KEYS,BIDDING_KEYS,RESPONSE_KEYS, execute_image_recognition_for_file,read_image_analysis_result
from RequestAndResponse import *
import asyncio
app = FastAPI(title="评分会话接口 API", debug=True)
review_system = ReviewSystem()
BASE_DIR = Path("./data/sessions")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# 方案1：使用后台线程池执行异步任务（推荐）
import concurrent.futures
import threading
from functools import partial

# 创建全局线程池和事件循环
_executor = None
_background_loop = None
_background_thread = None


# 方案1：使用后台线程池（推荐）
class BackgroundTaskManager:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.loop = None
        self.loop_thread = None
        self._start_background_loop()

    def _start_background_loop(self):
        """启动后台事件循环线程"""

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()

        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

        # 等待循环启动
        import time
        while self.loop is None:
            time.sleep(0.01)

    def submit_async_task(self, coro):
        """提交异步任务到后台循环"""
        if self.loop is None:
            raise RuntimeError("Background loop not running")

        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future

    def shutdown(self):
        """关闭任务管理器"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.executor:
            self.executor.shutdown(wait=True)


# 全局任务管理器
task_manager = BackgroundTaskManager()


def trigger_image_recognition_v1(session: Dict[str, Any], file_type: str, file_path: str, filename: str):
    """方案1：使用后台事件循环（推荐）"""
    try:
        file_name = Path(file_path).stem
        output_folder = f"./image_algorithm/output/{file_name}"

        # 提交异步任务到后台循环
        future = task_manager.submit_async_task(
            execute_image_recognition_for_file(file_path, output_folder, file_name)
        )

        # 创建任务信息
        task_info = {
            'future': future,
            'file_path': file_path,
            'file_name': file_name,
            'file_type': file_type,
            'output_folder': output_folder,
            'filename': filename,
            'status': 'running',
            'created_at': threading.current_thread().ident
        }

        # 初始化session中的图片任务列表
        if "image_tasks" not in session:
            session["image_tasks"] = []

        session["image_tasks"].append(task_info)
        session["image_recognition_started"] = True

        print(f"已为文件 {filename} ({file_type}) 启动图片识别任务")

        # 添加完成回调
        def on_complete(fut):
            try:
                result = fut.result()
                task_info['status'] = 'completed'
                task_info['result'] = result
                print(f"文件 {filename} 的图片识别任务已完成")
            except Exception as e:
                task_info['status'] = 'failed'
                task_info['error'] = str(e)
                print(f"文件 {filename} 的图片识别任务失败: {e}")

        future.add_done_callback(on_complete)

    except Exception as e:
        print(f"启动图片识别任务失败: {e}")

# Endpoints
@app.post("/create_session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    session_id = f"sess_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:6]}"
    session_path = BASE_DIR / session_id
    session_path.mkdir(parents=True)
    sessions[session_id] = {
        "path": session_path,
        "created_at": datetime.now().isoformat(),
        "files": {key: None for key in ALL_KEYS},
        "principles": {"sections":{}},
        "scores": {"weights":{},"sections":{},"type_score":{}},
        "total_score":{},
        "image_analysis_results":""
    }
    return SessionResponse(session_id=session_id)


# 方案1：在上传函数中直接触发图片识别（推荐）
@app.post("/upload_file", response_model=FileUploadResponse)
def upload_file(params: dict = Depends(get_upload_params)) -> FileUploadResponse:
    session_id = params["session_id"]
    file_type = params["file_type"]
    file = params["file"]

    validate_file_type(file_type)
    session = validate_session_exists(session_id)
    if session["files"][file_type] is not None:
        raise HTTPException(status_code=400, detail=f"File for type '{file_type}' already uploaded")
    if file_type in ["bidding_announcement", "technical_specification", "price_requirement"]:
        session["scores"] = {"weights": {}, "sections": {}, "type_score": {}}

    folder = session["path"] / file_type
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / file.filename
    with open(path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    info = {"file_type": file_type, "status": "uploaded", "name": file.filename, "size": path.stat().st_size}
    session["files"][file_type] = {**info, "path": str(path)}

    # 新增：如果是需要图片识别的文件类型，立即启动识别任务
    if file_type in ["business_response", "technical_response", "price_response"]:
        trigger_image_recognition_v1(session, file_type, str(path), file.filename)

    return FileUploadResponse(**info)


import asyncio
@app.post("/generate_principles")
async def generate_principles_stream(request: PrinciplesRequest = Body(...)) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        score_mapping = {
              "技术评分细则":"technical",
             "商务评分细则":"business" ,
             "价格评分细则":"price"
        }
        score_type = request.score_type

        session = validate_session_exists(request.session_id)

        if score_type in session["principles"]["sections"].keys():
                temp = session["principles"]["sections"][score_type]

                from function import async_line_chunks
                async for chunk in async_line_chunks(temp):
                    yield f"data: {chunk}\n\n"
                return

        if not session["files"].get("bidding_announcement"):
            yield f"data: {json.dumps({'status': 'error', 'message': '', 'data': 'Missing required [bidding_announcement]'}, ensure_ascii=False)}\n\n"
            return

        merged = Document()
        for key in BIDDING_KEYS:
            if info := session["files"].get(key):
                doc = Document(info["path"])
                for el in doc.element.body:
                    merged.element.body.append(el)
        merged_file = session["path"] / f"merged_{uuid4().hex}.docx"
        merged.save(str(merged_file))

        chapters = review_system.load_chapters(str(merged_file))
        # 流式查找评审原则
        chapter_name = None
        content = None
        all_chaps = None
        async for msg in review_system.find_review_principles_stream(chapters):
                # 转发查找进度
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg.get("status") == "complete" and msg.get("found"):
                    chapter_name = msg["chapter"]
                    content = msg["content"]
                    all_chaps = msg["all_chapters"]
                elif msg.get("status") == "complete" and not msg.get("found"):
                    yield f"data: {json.dumps({'status': 'error', 'data': '未找到评审原则章节'}, ensure_ascii=False)}\n\n"
                    return
        # 检查是否成功获取章节信息
        if not all_chaps or not chapter_name or not content:
            yield f"data: {json.dumps({'status': 'error', 'data': f'章节信息获取失败 - chapter_name: {chapter_name}, content: {bool(content)}, all_chaps: {all_chaps}'}, ensure_ascii=False)}\n\n"
            return

        async for msg in review_system.learn_review_rules_stream(content, all_chaps, agent=review_system.agent):
            # 如果是最后一条 complete，则写回 session
            if isinstance(msg, dict):
                if msg.get("status") == "end":
                    weights = msg["message"][0]
                    sections = msg["message"][1]
                    # 归一化
                    def remove_parentheses(text):
                        import re
                        pattern = r'[（(].*?[）)]'
                        if re.search(pattern, text):
                            # 有括号，去掉括号内容
                            return re.sub(pattern, '', text).strip()
                        else:
                            # 没有括号，返回原文本
                            return text.strip()
                    num = {remove_parentheses(k): float(v) for k, v in weights.items()}
                    tot = sum(num.values())
                    normalized = {score_mapping[k]: (v / 100 if tot == 100 else v) for k, v in num.items()}
                    session["principles"] = {
                        "chapter": chapter_name,
                        "weights": normalized,
                        "sections": sections
                    }

        if score_type in session["principles"]["sections"].keys():
            temp = session["principles"]["sections"][score_type]

            from function import async_line_chunks
            async for chunk in async_line_chunks(temp):
                yield f"data: {chunk}\n\n"
        else:
            yield f"data:{json.dumps({'status': 'error', 'message': '', 'data': '未查到相关评分细则'})}\n\n"
        yield f"data: {json.dumps({'status': '[DONE]', 'message': '[DONE]', 'data': '[DONE]'}, ensure_ascii=False)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/generate_score")
async def generate_score_stream(request: ScoreRequest = Body(...)) -> StreamingResponse:
    async def event_stream() -> AsyncGenerator[str, None]:
        session = validate_session_exists(request.session_id)
        if not session["files"].get("business_response") or not session["files"].get("technical_response"):
            yield f"data: {json.dumps({'status': 'error', 'message': 'Missing required responses'}, ensure_ascii=False)}\n\n"
            return
        score_type = request.score_type
        enable_image_recognition = getattr(request, 'enable_image_recognition', True)
        score_info = session["principles"]
        if not score_info:
            yield f"data: {json.dumps({'status': 'error', 'message': '评分标准未生成'}, ensure_ascii=False)}\n\n"
            return

        chapter = score_info["chapter"]
        weights = score_info["weights"]
        temp_folder = session["path"] / f"response_folder_{uuid4().hex}"
        temp_folder.mkdir()

        for key in RESPONSE_KEYS:
            if info := session["files"].get(key):
                shutil.copy(info["path"], temp_folder / Path(info["path"]).name)
                # 在开头启动图片识别任务（不等待）
                yield f"data: {json.dumps({'status': 'info', 'message': '启动多文件图片识别任务'}, ensure_ascii=False)}\n\n"

        yield f"data: 开始评分：{chapter}\n\n"

        total_score = 0.0
        all_details= []

        score_mapping = {
            "technical": "technical_response",
            "business": "business_response",
            "price": "price_response"
        }
        sec = str()
        w = str()
        # 新增：用于收集所有评分消息
        for sec1, w1 in weights.items():
            if score_mapping[sec1] == score_type:
                sec = sec1
                w = w1
                break
        if sec == '':
            yield f"data: {json.dumps({'status': 'error', 'message': '评分标准未生成'}, ensure_ascii=False)}\n\n"
        async for msg in review_system.score_document_stream(
                str(temp_folder), chapter, sec, {sec: w}
        ):
            # msg 是字典，包含 status, score, section 等字段
            # 如果是 section_done，则累加得分并收集细节
            if msg.get("status") == "section_done":
                total_score += msg.get("data", 0)
                all_details = msg.get("message", {})
                session["total_score"][sec] = total_score
            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

        # 根据参数决定是否进行图片识别处理
        if enable_image_recognition:
            # 必须等待图片识别任务完成
            image_analysis_results = {}
            try:
                if not session.get("image_recognition_started", False):
                    yield f"data: {json.dumps({'status': 'info', 'message': '没有图片识别任务需要等待'})}\n\n"
                else:
                    image_tasks = session.get("image_tasks", [])
                    if image_tasks:
                        if not session["image_analysis_results"]:
                            yield f"data: {json.dumps({'status': 'info', 'message': f'等待 {len(image_tasks)} 个图片识别任务完成'}, ensure_ascii=False)}\n\n"

                            # 等待所有图片识别任务完成
                            completed_tasks = []
                            for task_info in image_tasks:
                                file_name = task_info['file_name']
                                filename = task_info.get('filename', file_name)
                                try:
                                    # 方案1和方案2：使用 future 的情况
                                    if 'future' in task_info:
                                        future = task_info['future']
                                        # 在异步环境中等待 future 完成
                                        def wait_for_future():
                                            try:
                                                return future.result(timeout=300)  # 5分钟超时
                                            except concurrent.futures.TimeoutError:
                                                raise Exception("图片识别任务超时（5分钟）")
                                            except Exception as e:
                                                raise Exception(f"图片识别任务执行失败: {str(e)}")
                                        loop = asyncio.get_event_loop()
                                        await loop.run_in_executor(None, wait_for_future)
                                        completed_tasks.append(task_info)
                                        yield f"data: {json.dumps({'status': 'info', 'message': f'文件 {filename} 图片识别完成'})}\n\n"
                                except Exception as e:
                                    import traceback
                                    error_traceback = traceback.format_exc()
                                    yield f"data: {json.dumps({'status': 'warning', 'message': f'文件 {filename} 图片识别失败: {str(e)}'})}\n\n"
                                    print(f"图片识别任务异常: {error_traceback}")

                            # 读取所有图片识别结果
                            for task_info in completed_tasks:
                                try:
                                    file_key = task_info.get('file_type', task_info['file_name'])
                                    output_folder = task_info['output_folder']
                                    print(f"正在读取文件 {task_info['file_name']} 的图片分析结果，路径：{output_folder}")
                                    result = await read_image_analysis_result(output_folder)
                                    temp = task_info['file_name']
                                    if result:
                                        image_analysis_results[file_key] = {'file_name': task_info['file_name'], 'file_path': task_info['file_path'], 'analysis_result': {k: "".join(v) if isinstance(v, list) else v for k, v in result.items()}}
                                        yield f"data: {json.dumps({'status': 'info', 'message': f'成功读取文件 {temp} 的分析结果'}, ensure_ascii=False)}\n\n"
                                    else:
                                        yield f"data: {json.dumps({'status': 'warning', 'message': f'文件 {temp} 没有分析结果'}, ensure_ascii=False)}\n\n"
                                except Exception as e:
                                    import traceback
                                    error_traceback = traceback.format_exc()
                                    temp = task_info['file_name']
                                    yield f"data: {json.dumps({'status': 'warning', 'message': f'读取文件 {temp} 分析结果失败: {str(e)}'}, ensure_ascii=False)}\n\n"
                                    print(f"读取分析结果异常: {error_traceback}")

                            # 如果有图片分析结果，进行大模型分析
                            if image_analysis_results:
                                yield f"data: {json.dumps({'status': 'info', 'message': f'开始大模型分析 {len(image_analysis_results)} 个文件的图片内容'}, ensure_ascii=False)}\n\n"
                                session["image_analysis_results"] = image_analysis_results
                        print(image_analysis_results)
                        async for msg in review_system.process_multiple_image_analysis_with_llm_stream(
                                image_analysis_results, all_details, w):
                            if msg.get("status") == "ImageScore_Done":
                                total_score = msg.get("data", 0)
                                all_details = msg.get("message", {})
                                session["total_score"][sec] = total_score
                            yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"

                        yield f"data: {json.dumps({'status': 'info', 'message': '大模型图片分析完成'})}\n\n"

                    else:
                        yield f"data: {json.dumps({'status': 'info', 'message': '没有找到图片识别任务'})}\n\n"
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                yield f"data: {json.dumps({'status': 'error', 'message': f'图片识别处理失败: {str(e)}'})}\n\n"
                print(f"图片识别等待异常: {error_traceback}")

        session["scores"]["weights"][sec] = w
        session["scores"]["sections"][sec] = all_details
        session["scores"]["type_score"][sec] = total_score
        yield f"data: {json.dumps({'status': 'complete', 'message': all_details, 'data': total_score}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'status': '[DONE]', 'message': '[DONE]', 'data': '[DONE]'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/list_sessions", response_model=List[SessionListItem])
def list_sessions() -> List[SessionListItem]:
    return [SessionListItem(
        session_id=sid,
        created_at=info["created_at"],
        file_count=sum(1 for v in info["files"].values() if v)
    ) for sid, info in sessions.items()]


@app.post("/scoring", response_model=float)
def scoring(request: ScoringRequest) -> float:
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session_info = sessions[session_id]
    score = 0.0
    for key in session_info["total_score"].keys():
        score += session_info["total_score"][key]
    return score

@app.post("/session_detail")
def session_detail(request: SessionDetailRequest = Body(...)) -> Dict[str, Any]:
    session_data = validate_session_exists(request.session_id)
    field = request.field
    if field == "path":
        raise HTTPException(status_code=403, detail="Access to path field is forbidden")
    if field is None:
        full_data = session_data.copy()
        full_data["files"] = {k: {"name": v["name"], "size": v["size"]} if v else None for k, v in full_data["files"].items()}
        full_data["available_fields"] = list(session_data.keys())
        return full_data
    if field not in session_data:
        available_keys = list(session_data.keys())
        raise HTTPException(status_code=404, detail=f"Field '{field}' not found. Available_FIELDS: {', '.join(available_keys)}")
    if field == "files":
        return {"files": {k: {"name": v["name"], "size": v["size"]} if v else None for k, v in session_data["files"].items()}}
    return {field: session_data[field]}

@app.post("/delete_file", response_model=DeleteResponse)
def delete_file(request: DeleteFileRequest = Body(...)) -> DeleteResponse:
    validate_session_exists(request.session_id)
    validate_file_type(request.file_type)
    fi = sessions[request.session_id]["files"].get(request.file_type)
    if not fi:
        raise HTTPException(status_code=404, detail="File not found")
    try:
        os.remove(fi["path"])
    except:
        pass
    sessions[request.session_id]["files"][request.file_type] = None
    return DeleteResponse(status="deleted", file_type=request.file_type)

@app.post("/delete_session", response_model=DeleteResponse)
def delete_session(request: DeleteSessionRequest = Body(...)) -> DeleteResponse:
    session = validate_session_exists(request.session_id)
    shutil.rmtree(session["path"], ignore_errors=True)
    del sessions[request.session_id]
    return DeleteResponse(status="deleted", session_id=request.session_id)

