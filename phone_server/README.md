# Phone Server

Servidor HTTP simples para expor APIs do Termux:API (Galaxy A30S + Termux:API).

## Como rodar no Termux

1. Defina o token de acesso como variável de ambiente:
   ```bash
   export DAVIOS_PHONE_TOKEN="seu_token_aqui"
   ```
   (O token deve ser o mesmo configurado no lado do assistente.)

2. Opcionalmente, ajuste host e porta:
   ```bash
   export DAVIOS_PHONE_HOST="0.0.0.0"  # ou deixe padrão 127.0.0.1
   export DAVIOS_PHONE_PORT="8765"
   ```

3. Execute o servidor:
   ```bash
   python3 phone_server.py
   ```

O servidor ficará escutando em `http://$DAVIOS_PHONE_HOST:$DAVIOS_PHONE_PORT`.

## Como testar do PC (com runner falso)

Para testes, use o arquivo `tests/test_phone_server.py`. Ele inicia o servidor em uma porta efêmera com um runner falso que simula os comandos do Termux.

Execute:
```bash
pytest tests/test_phone_server.py -v
```

## Endpoints

Todos os endpoints requerem o header `Authorization: Bearer <token>`.

- `GET /battery` → nível de bateria e status de carregamento
- `GET /location` → localização GPS (latitude, longitude, accuracy)
- `GET /device` → modelo do dispositivo e versão do Android
- `POST /tts/speak` → faz o falar o texto fornecido (máximo 300 caracteres)
- `POST /vibrate` → vibra o dispositivo por N milissegundos (50–2000 ms)

Consulte o código-fonte para detalhes de validação e tratamento de erros.