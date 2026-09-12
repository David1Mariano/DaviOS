 brain/action_manager.py           |  74 [32m+++++++++++[m
 brain/cognitive_core.py           |  11 [32m+[m[31m-[m
 brain/prompt_builder.py           |  80 [32m+++++++++++[m[31m-[m
 brain/tool_execution_flow.py      |  68 [32m+++++++[m[31m---[m
 brain/tool_request_parser.py      |  58 [32m+++++++++[m
 config/davios.json                |   7 [32m+[m[31m-[m
 config/davios_config.py           |  12 [32m++[m
 main.py                           |  10 [32m++[m
 patch_sanitize.py                 |  24 [32m++++[m
 personality/personality.py        |   1 [32m+[m
 tests/test_d1_fewshot.py          |  37 [32m++++++[m
 tests/test_d1_read_file.py        |  63 [32m++++++++++[m
 tests/test_prompt_tools.py        | 194 [32m+++++++++++++++++++++++++++[m[31m--[m
 tests/test_tool_execution.py      |   2 [32m+[m[31m-[m
 tests/test_tool_execution_flow.py |  93 [32m+++++++++++++[m[31m-[m
 tests/test_tool_request_parser.py |  93 [32m+++++++++++++[m[31m-[m
 tmp/baseline.txt                  |   4 [31m-[m
 tmp/collect.txt                   | 222 [31m---------------------------------[m
 tmp/diffstat.txt                  |  16 [31m---[m
 tmp/e2e_final.txt                 |  24 [31m----[m
 tmp/e2e_result.txt                |  29 [31m-----[m
 tmp/e2e_result2.txt               |  24 [31m----[m
 tmp/e2e_v2.txt                    |  11 [31m--[m
 tmp/e2e_v3.txt                    |  14 [31m---[m
 tmp/e2e_v4.txt                    |  38 [31m------[m
 tmp/offline_test.txt              |   2 [31m-[m
 tmp/pytest_all.txt                | Bin [31m530[m -> [32m0[m bytes
 tmp/pytest_clean.txt              |   2 [31m-[m
 tmp/pytest_final.txt              | 254 [31m--------------------------------------[m
 tmp/pytest_full.txt               | Bin [31m11522[m -> [32m0[m bytes
 tmp/pytest_integr.txt             |  92 [31m--------------[m
 tmp/pytest_integr2.txt            |  10 [31m--[m
 tmp/pytest_integr_final.txt       |  16 [31m---[m
 tmp/pytest_integr_final2.txt      |   2 [31m-[m
 tmp/pytest_partb.txt              | Bin [31m530[m -> [32m0[m bytes
 tmp/pytest_result.txt             |   4 [31m-[m
 tmp/pytest_resume.txt             |  17 [31m---[m
 tmp/pytest_run.txt                | Bin [31m4368[m -> [32m0[m bytes
 tmp/pytest_run2.txt               | Bin [31m5354[m -> [32m0[m bytes
 tmp/pytest_run3.txt               |  17 [31m---[m
 tmp/pytest_run4.txt               |  31 [31m-----[m
 tmp/pytest_run5.txt               |   2 [31m-[m
 tmp/pytest_status.txt             |  66 [31m----------[m
 tmp/pytest_style.txt              |   4 [31m-[m
 tmp/test_current.txt              |   4 [31m-[m
 45 files changed, 791 insertions(+), 941 deletions(-)
