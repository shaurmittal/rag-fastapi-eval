"""Stage 2 of curation: the human-reviewed golden set.

Every entry below was selected by reading the issue and its chosen answer in
data/eval_candidates.jsonl, then checking the pinned corpus (v0.115.0) for
the file that grounds the answer - with include directives resolved, since
that's the text that gets indexed.

Two separate judgments per entry:
  - The QUESTION comes from GitHub, verbatim (cleaned of template noise). That
    keeps the query distribution authentic.
  - The EXPECTED ANSWER is written against the pinned corpus, not copied from
    the thread. Several threads predate the feature that now answers them
    (#1708 include_in_schema, #1989 form models, #4734 return-type response
    models) - the question is valid ground truth even though the historical
    reply is a proposal or a workaround.

expected_sources were found by grepping the corpus for the APIs the answer
names - never by running our own retriever, which would let the system under
test choose its own ground truth.

question_type is where the answer primarily lives: "docs" or "code".

RUN
    uv run python -m mine_eval_set.curated     # writes data/golden_eval_set.jsonl
"""

import json

CANDIDATES = "data/eval_candidates.jsonl"
OUT = "data/golden_eval_set.jsonl"

D = "docs/en/docs/"

GOLDEN = [
    (3361, "docs", [D + "tutorial/handling-errors.md"],
     "Override the RequestValidationError handler with @app.exception_handler(RequestValidationError). "
     "The exception's .errors() holds the detailed validation errors (and .body the received body), "
     "so the handler can log them before returning a 422 JSONResponse."),
    (1294, "docs", [D + "advanced/behind-a-proxy.md"],
     "Use root_path: start Uvicorn with --root-path /api/1.0 (or pass root_path to FastAPI) so the app "
     "knows the prefix the proxy adds; the docs UI then requests the OpenAPI schema with that prefix."),
    (3, "docs", [D + "tutorial/response-status-code.md", D + "tutorial/background-tasks.md"],
     "Set status_code=202 in the path operation decorator, and schedule the long work with a "
     "BackgroundTasks parameter (background_tasks.add_task), which runs after the response is sent."),
    (59, "docs", [D + "tutorial/background-tasks.md"],
     "Declare a parameter of type BackgroundTasks in the path operation function and call "
     "background_tasks.add_task(func, *args). FastAPI runs the task after returning the response."),
    (852, "docs", [D + "advanced/using-request-directly.md"],
     "Form() can't declare an arbitrary dict of fields. Declare a Request parameter and read the form "
     "directly with await request.form()."),
    (896, "docs", [D + "tutorial/body-multiple-params.md"],
     "With a single body model, FastAPI expects the model's fields at the root of the JSON. Use "
     "Body(embed=True) to expect the model nested under a key named after the parameter."),
    (1708, "docs", [D + "tutorial/query-params-str-validations.md"],
     "Pass include_in_schema=False to Query (or Header, Cookie, etc.) to exclude that parameter from "
     "the generated OpenAPI schema and the docs UI."),
    (246, "docs", [D + "tutorial/path-params.md"],
     "Declare the parameter's type as a str-based Enum. The allowed values are then part of the schema "
     "and shown as a dropdown in the docs UI."),
    (280, "docs", [D + "tutorial/schema-extra-example.md"],
     "Declare examples on the Pydantic model itself, via model_config = {'json_schema_extra': "
     "{'examples': [...]}} or Field(examples=[...]); they appear in the JSON Schema and the docs."),
    (2586, "docs", [D + "advanced/response-directly.md", D + "advanced/response-change-status-code.md"],
     "Return a Response such as JSONResponse(status_code=404, content=...) directly, or declare a "
     "Response parameter and set response.status_code."),
    (528, "docs", [D + "tutorial/response-model.md"],
     "Use response_model_exclude_none=True (or response_model_exclude_unset=True) in the path "
     "operation decorator to omit None / unset fields from the response."),
    (1693, "docs", [D + "tutorial/dependencies/index.md"],
     "Pass the dependency function itself to Depends without calling it: Depends(get_db), not "
     "Depends(get_db()). FastAPI calls it for you."),
    (19, "docs", [D + "advanced/using-request-directly.md"],
     "Declare a path operation parameter annotated with Request (from fastapi); FastAPI passes the "
     "request object in."),
    (86, "docs", [D + "tutorial/extra-models.md"],
     "Use a Union of the models (rendered as anyOf in OpenAPI). Put the more specific type first, "
     "since Pydantic tries the types in order."),
    (753, "docs", [D + "tutorial/bigger-applications.md",
                   D + "tutorial/dependencies/dependencies-in-path-operation-decorators.md"],
     "Pass dependencies=[Depends(...)] to APIRouter(...) or include_router(...) and they run for every "
     "path operation in that router. Use decorator-level dependencies when the return value isn't needed."),
    (1821, "docs", [D + "tutorial/request-files.md"],
     "The parameters are UploadFile objects, not paths. Read their content with await file.read() "
     "(or file.file.read() in sync code) and use file.filename for the name."),
    (124, "docs", [D + "tutorial/response-model.md"],
     "The function may return a UserInDB, but response_model=User filters the output to the fields of "
     "User, so hashed_password is not included in the response."),
    (23, "docs", [D + "tutorial/debugging.md"],
     "Import uvicorn and call uvicorn.run(app, host=..., port=...) inside if __name__ == '__main__':, "
     "then run or debug that file directly from the editor."),
    (830, "docs", [D + "tutorial/cors.md"],
     "Add CORSMiddleware with allow_origins set to the list of exact origins, and allow_methods / "
     "allow_headers given as lists of strings (for example ['*'])."),
    (747, "docs", [D + "tutorial/dependencies/dependencies-in-path-operation-decorators.md"],
     "dependencies=[Depends(...)] belongs in the path operation decorator (@app.get(..., "
     "dependencies=[...])), not in the function signature."),
    (873, "docs", [D + "tutorial/body-nested-models.md"],
     "Declare the body as a dict, e.g. dict[str, Any] or Dict[str, str], to accept keys you don't "
     "know in advance (bodies of arbitrary dicts)."),
    (5420, "docs", [D + "tutorial/path-params.md"],
     "Use a path convertor: a route like /aaa/{file_path:path} captures the rest of the path, "
     "slashes included."),
    (1093, "docs", [D + "tutorial/request-forms.md"],
     "Receiving form data (and files) requires the python-multipart package: pip install "
     "python-multipart."),
    (1184, "docs", [D + "advanced/testing-dependencies.md"],
     "Set app.dependency_overrides[original_dependency] = override in the test, and reset it with "
     "app.dependency_overrides = {} (or pop the key) afterwards."),
    (945, "docs", [D + "tutorial/path-params-numeric-validations.md"],
     "A path parameter is always required, since it's part of the path. Make the value optional with "
     "a separate route or by using a query parameter instead."),
    (3220, "docs", [D + "advanced/sub-applications.md"],
     "Create separate FastAPI applications and mount them with app.mount('/subapi', subapi). Each "
     "mounted sub-application has its own independent /docs."),
    (1650, "code", ["fastapi/routing.py"],
     "APIRouter(...) and include_router(...) accept include_in_schema=False, which excludes every "
     "route in that router from the OpenAPI schema."),
    (1173, "code", ["fastapi/openapi/utils.py", D + "how-to/extending-openapi.md"],
     "Import the app and call app.openapi() (built on fastapi.openapi.utils.get_openapi, which takes "
     "title, version and routes) and write the result to a JSON file - no running server needed."),
    (1989, "docs", [D + "tutorial/request-form-models.md"],
     "Since FastAPI 0.113.0 you can declare a Pydantic model as a form: Annotated[FormData, Form()], "
     "and FastAPI extracts the form fields into the model."),
    (2188, "code", ["fastapi/routing.py", "fastapi/applications.py"],
     "Use @app.api_route('/items/', methods=['GET', 'POST']) (also on APIRouter) to register one "
     "function for several HTTP methods, or stack @app.get and @app.post decorators."),
    (27, "docs", [D + "tutorial/dependencies/dependencies-with-yield.md", D + "tutorial/sql-databases.md"],
     "Create one session per request in a dependency that yields it, and close it after the yield "
     "(try/finally); inject it into path operations with Depends."),
    (1280, "docs", [D + "async.md"],
     "Use async def and await an async client (such as httpx.AsyncClient) for concurrent I/O. Use "
     "plain def for blocking libraries; FastAPI runs def path operations in a threadpool."),
    (612, "code", ["fastapi/security/http.py", D + "tutorial/security/first-steps.md"],
     "OpenAPI doesn't allow declaring Authorization as an ordinary header parameter. Use a security "
     "scheme such as HTTPBearer from fastapi.security, which reads the Authorization header and adds "
     "an Authorize button to the docs."),
    (334, "docs", [D + "advanced/custom-response.md"],
     "Blocking calls inside an async generator block the event loop. StreamingResponse also accepts a "
     "normal (sync) generator or iterator, which avoids blocking the loop."),
    (22, "docs", [D + "advanced/websockets.md"],
     "Declare @app.websocket('/ws') with a WebSocket parameter, await websocket.accept(), then loop on "
     "websocket.receive_text() / send_text()."),
    (110, "docs", [D + "how-to/custom-docs-ui-assets.md", "fastapi/openapi/docs.py"],
     "Disable the default docs (docs_url=None), serve the Swagger UI JavaScript and CSS yourself with "
     "StaticFiles, and add a /docs route returning get_swagger_ui_html(...) pointing at those files."),
    (154, "docs", [D + "advanced/response-cookies.md"],
     "Declare a Response parameter and call response.set_cookie(key=..., value=...), or create and "
     "return a Response directly after setting the cookie on it."),
    (263, "docs", [D + "tutorial/query-params-str-validations.md"],
     "Declare the list parameter with Query and a default list, e.g. q: list[str] = Query(default="
     "['foo', 'bar']); Query must be used explicitly for list query parameters."),
    (321, "docs", [D + "tutorial/query-params-str-validations.md"],
     "A list-typed parameter would otherwise be read as a request body. Declare it explicitly with "
     "Query() to make it a query parameter."),
    (409, "docs", [D + "tutorial/middleware.md"],
     "Use @app.middleware('http') with a function taking (request, call_next): code before await "
     "call_next(request) runs before the path operation, code after it runs on the response."),
    (660, "code", ["fastapi/openapi/utils.py", D + "how-to/extending-openapi.md"],
     "The default 422 schema is added automatically from validation_error_definition / "
     "validation_error_response_definition in fastapi.openapi.utils. To change what the docs show, "
     "override app.openapi (extending OpenAPI) or document responses with the responses parameter."),
    (1072, "docs", [D + "advanced/testing-events.md"],
     "Use TestClient as a context manager, with TestClient(app) as client:, so the startup and "
     "shutdown (lifespan) events run."),
    (1216, "docs", [D + "how-to/custom-request-and-route.md", D + "tutorial/handling-errors.md"],
     "For validation errors, RequestValidationError exposes the received body as exc.body. More "
     "generally, a custom APIRoute class can capture the request body for use in exception handling."),
    (1480, "docs", [D + "advanced/events.md"],
     "That's expected: lifespan events (startup and shutdown) only run for the main application, not "
     "for mounted sub-applications."),
    (1544, "docs", [D + "how-to/configure-swagger-ui.md"],
     "Pass swagger_ui_parameters={'docExpansion': 'none'} to FastAPI(...) to collapse the "
     "operations by default."),
    (2765, "docs", [D + "advanced/settings.md"],
     "Using a settings class directly in Depends makes FastAPI treat its fields as query parameters. "
     "Create a get_settings() function (cached with @lru_cache) that returns Settings(), and depend on it."),
    (3720, "docs", [D + "advanced/events.md"],
     "Use the lifespan parameter: an async context manager passed to FastAPI(lifespan=...), whose "
     "code before the yield runs once at startup."),
    (4448, "docs", [D + "tutorial/handling-errors.md"],
     "HTTPException has to be raised, not returned. Returning it serializes it as ordinary data in a "
     "200 response."),
    (4734, "docs", [D + "tutorial/response-model.md"],
     "FastAPI uses the function's return type annotation as the response model when response_model "
     "isn't set, so -> list[Item] gives validation, filtering and documentation."),
    (5611, "docs", [D + "tutorial/body-multiple-params.md"],
     "Make the body optional by giving it a default of None: request: RegistrationRequest | None = None."),
    (2222, "code", ["fastapi/encoders.py"],
     "jsonable_encoder takes a custom_encoder argument, a dict mapping types to functions, e.g. "
     "custom_encoder={np.ndarray: lambda a: a.tolist()}."),
]


def main():
    candidates = {c["issue"]: c for c in map(json.loads, open(CANDIDATES))}
    with open(OUT, "w") as f:
        for issue, qtype, sources, answer in GOLDEN:
            c = candidates[issue]
            f.write(
                json.dumps(
                    {
                        "id": f"gh-{issue}",
                        "question": c["question"],
                        "expected_answer": answer,
                        "expected_sources": sources,
                        "question_type": qtype,
                        "source_url": c["url"],
                    }
                )
                + "\n"
            )
    n_code = sum(1 for g in GOLDEN if g[1] == "code")
    print(f"wrote {len(GOLDEN)} examples to {OUT}  (docs {len(GOLDEN) - n_code}, code {n_code})")


if __name__ == "__main__":
    main()
