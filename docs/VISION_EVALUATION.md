# Evaluacion de Vision Computacional (Parte B.3)

Estado: **EVALUACION SOLO** (nada implementado en esta etapa).

## Resumen

El Qwen3-4B-Q4_K_M actual (GGUF en llama.cpp) es un modelo **texto-only**:
no acepta imagenes como input. Ejecutar vision sobre el mismo simplemente
no es posible con el backend actual.

## Opciones evaluadas

| Opcion | Soporta vision | Cambio necesario | Coste |
|--------|---------------|------------------|-------|
| Qwen3-4B-Q4_K_M actual | No | Ninguno | 0 GB |
| Qwen2.5-VL-7B / Qwen2-VL (GGUF) | Si (multimodal) | llama.cpp con soporte de proyeccion de vision + modelo mas grande | ~5-7 GB VRAM/RAM |
| modelador auxiliar separado (ej. CLIP para embeddings de imagen) | Si (solo embeddings, no conversacion visual) | Modelo auxiliar aparte + integracion de tool | ~1-2 GB |
| API remota (OLMo/Qwen-VL en nube) | Si | Viola offline-first; descartada | - |

## Requisito de hardware local

- GPU AMD RX 6600: 8 GB VRAM. Un Qwen2.5-VL Q4 cabría en VRAM, pero la
  integración con llama.cpp standalone requiere:
  1. `llama-server` con soporte de proyeccion multimodal (`--mmproj` y
     GGUF del vision tower);
  2. modelo visual descargado (fuera de esta etapa);
  3. pipeline de "imagen -> text prompt" o conversacion con imagen.

## Recomendacion (futura, fuera de alcance ahora)

1. **No mezclar** vision en el Qwen3-4B de texto.
2. Como siguiente paso de vision real, usar **Qwen2.5-VL-7B Q4** mediante
   llama.cpp con `--mmproj` cuando la RAM lo permita (16 GB RAM total en
   esta maquina = ajustado pero viable con Q4).
3. Alternativa ligera: modelo auxiliar de embeddings de imagen (CLIP) para
   **buscar en memoria por similitud visual**, sin conversacion de imagen.

## Decision de esta etapa

NO implementar vision. La funcionalidad de "ver imagenes" no se anuncia al
usuario; DaviOS responde que aun no dispone de esa capacidad.