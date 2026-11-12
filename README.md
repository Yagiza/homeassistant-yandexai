See [description in English](#yandex-ai-integration-for-home-assistant) below 👇
<br>
<br>

# Интеграция Yandex AI для Home Assistant

[![Добавить репозиторий в HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Yagiza&repository=homeassistant-yandexai&category=integration) [![Настроить интеграцию с Yandex AI](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=yandex_ai)

Ассистент с искусственным интеллектом для вашего умного дома. Интеграция объединяет возможности [Yandex AI Studio](https://yandex.cloud/ru/docs/ai-studio/) с Home Assistant, позволяя создать полноценную систему управления умным домом с естественным языковым интерфейсом.

Данная интеграция является форком [интеграции YandexGPT](https://github.com/black-roland/homeassistant-yandexgpt). Код оригинальной интеграции был полностью переработан: вместо устаревшего Yandex AI Studio SDK используется OpenAI-совместимый API. Основной целью создания данной интеграции стала поддержка [AI task](https://www.home-assistant.io/integrations/ai_task/) — новой функции Home Assistant.

## Возможности

- Общение с ассистентом через интерфейс Home Assistant.
- Полноценное управление умным домом: ассистент может не только сообщать состояние устройств, но и управлять ими (включать свет, регулировать температуру и т.д.), а также запускать скрипты и автоматизации ([демо](https://boosty.to/mansmarthome/posts/bb1e2d91-6edb-4dfe-b96f-99de636ce844)).
- Общение с ассистентом через приложение Home Assistant на смартфоне или умных часах.
- Интеграция может служить «мозгом» для создания DIY-умной колонки на базе ESPHome, если помимо Yandex AI добавить в Home Assistant синтез и распознавание речи.
- Yandex AI можно использовать в автоматизациях, например, для создания чат-бота в Telegram.
- Поддержка моделей Yandex AI Studio через OpenAI-совместимый API: Alice AI, YandexGPT и сторонние модели (Qwen и DeepSeek); также можно ввести любой идентификатор модели.
- Можно создать несколько ассистентов с разными моделями и настройками, а также использовать Yandex AI в [ИИ-задачах](https://www.home-assistant.io/integrations/ai_task/) (извлечение данных, анализ текста и т.д.).
- Генерация изображений в [ИИ-задачах](https://www.home-assistant.io/integrations/ai_task/) с помощью Alice AI Image Art 3.0 и Yandex Art.
- **Анализ изображений** в [ИИ-задачах](https://www.home-assistant.io/integrations/ai_task/) с помощью мультимодальных моделей DeepSeek и Qwen.

Yandex AI Studio — это облачный сервис, плата за который взимается в соответствии с тарифами Yandex Cloud. При первой регистрации [можно получить стартовый грант](https://yandex.cloud/ru/docs/getting-started/usage-grant).

## Установка и настройка

Инструкции по получению ключа API и настройке интеграции можно найти в [wiki](https://github.com/Yagiza/homeassistant-yandexai/wiki).

TLDR: Добавьте интеграцию, используя голубые кнопки выше, а затем получите [идентификатор каталога](https://yandex.cloud/ru/docs/resource-manager/operations/folder/get-id) и [ключ API](https://yandex.cloud/ru/docs/iam/operations/api-key/create). Сервисному аккаунту нужно указать роль `ai.languageModels.user`.

## Примеры использования

Примеры использования можно найти в [блоге автора оригинальной интеграции](https://mansmarthome.info/tags/yandexgpt/). Кроме того, про первую версию интеграции [автор оригинальной интеграции](https://github.com/black-roland) рассказывал на [YouTube](https://www.youtube.com/watch?v=C1KcW--vnUo).

<p>
  <img src="https://github.com/user-attachments/assets/c4f2520d-a1e7-433b-99d6-9db29b2c99f1" height="340px" alt="Assist" />
  <img src="https://github.com/user-attachments/assets/34f05829-7a10-4087-8596-5087b8310533" height="340px" alt="Morning digests" />
</p>

## Генерация изображений

Сущность AI Task умеет генерировать изображения моделями Alice AI Image Art 3.0 и Yandex Art. По умолчанию создаётся ИИ-задача `Alice AI Image Art 3.0 Task`:

```yaml
service: ai_task.generate_image
data:
  task_name: "Котик"
  instructions: "Милый кот сидит на подоконнике, акварель"
  entity_id: ai_task.alice_ai_image_art_3_0_task  # по умолчанию
```

Изображение сохраняется в [медиа-источник ИИ-задач](https://www.home-assistant.io/integrations/ai_task/) (`media-source://ai_task/image/...`), сервис возвращает `media_source_id` и ссылку `url` (действует 1 час). Формат — JPEG, подсказка — только текст. Сервисному аккаунту нужна роль `ai.imageGeneration.user`.

## Альтернативы

Если вам нужны модели другого провайдера (например, Cloud.ru), рассмотрите интеграцию [Cloud.ru Foundation Models](https://github.com/black-roland/homeassistant-cloud-ru-ai). Для некоторых сценариев использования она может быть более экономичным вариантом.

Обе интеграции позволяют создавать умных ассистентов для Home Assistant, и [примеры использования из блога автора оригинальной интеграции](https://mansmarthome.info/tags/ai/) легко адаптируются для любой из них.

## Спасибо

Интеграция оказалась полезной? Хотите сказать спасибо? Кофе автору оригинальной интеграции — ваша благодарность ему. <kbd>[☕ На кофе](https://mansmarthome.info/donate/?utm_source=github&utm_medium=referral&utm_campaign=gpt#%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC%D0%B0-%D0%B1%D1%8B%D1%81%D1%82%D1%80%D1%8B%D1%85-%D0%BF%D0%BB%D0%B0%D1%82%D0%B5%D0%B6%D0%B5%D0%B9)</kbd> У меня нет учётных записей в сервисах доната, поэтому меня можно поблагодарить просто поделившись ссылкой на данную интеграцию с друзьями и знакомыми. Мне будет приятно осознавать, что кто-то пользуется плодами моего труда.

Большое спасибо всем, кто поддерживает авторов обеих интеграций:

![Спасибо][donors-list]

## Уведомление

Это независимая интеграция, разработанная сообществом. Данная интеграция не связана с Яндексом или Yandex Cloud. Yandex AI Studio — это сервис Yandex Cloud.

Данная интеграция не является официальным продуктом Яндекса и не поддерживается Яндексом.

---

# Yandex AI integration for Home Assistant

[![Add custom repository to HACS](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=Yagiza&repository=homeassistant-yandexai&category=integration) [![Set up Yandex AI integration](https://my.home-assistant.io/badges/config_flow_start.svg)](https://my.home-assistant.io/redirect/config_flow_start/?domain=yandex_ai)

AI-powered assistant for your smart home. This integration bridges [Yandex AI Studio](https://yandex.cloud/en/docs/ai-studio/) capabilities with Home Assistant, enabling a complete smart home control system with natural language interface.

This integration is a fork of the [YandexGPT integration](https://github.com/black-roland/homeassistant-yandexgpt). The original codebase has been completely rewritten: instead of the obsolete Yandex AI Studio SDK, it uses the OpenAI-compatible API. The main goal of this fork was to support [AI task](https://www.home-assistant.io/integrations/ai_task/), a new Home Assistant feature.

## Features

- Chat with the assistant directly from Home Assistant's interface.
- Full smart home control: the assistant can not only report device states but also control them (turn lights on/off, adjust temperature, etc.) and execute scripts/automations.
- Chat with the assistant using Home Assistant app on a smartphone or smartwatch.
- Can serve as the "brain" for a DIY smart speaker using ESPHome when combined with speech synthesis/recognition.
- Useful for creating Telegram chatbots and other automations.
- Supports Yandex AI Studio models via the OpenAI-compatible API: Alice AI, YandexGPT, and third-party models (Qwen and DeepSeek); any model id can be entered.
- You can create multiple agents with different models and settings, and use Yandex AI in [AI tasks](https://www.home-assistant.io/integrations/ai_task/) (data extraction, text analysis, etc.).
- Image generation in [AI tasks](https://www.home-assistant.io/integrations/ai_task/) with Alice AI Image Art 3.0 and Yandex Art.
- **Image analysis** in [AI tasks](https://www.home-assistant.io/integrations/ai_task/) with multimodal DeepSeek and Qwen models.

Yandex AI Studio is a cloud service with pricing according to Yandex Cloud tariffs.

## Set up

Use blue buttons above to install and configure the integration. Please check out the official documentation on how to retrieve [folder ID](https://yandex.cloud/en/docs/resource-manager/operations/folder/get-id) and [API key](https://yandex.cloud/en/docs/iam/operations/api-key/create). For a service account it's required to set the following role: `ai.languageModels.user`.

## Image generation

The AI Task entity can generate images with Alice AI Image Art 3.0 and Yandex Art. The default AI task is named `Alice AI Image Art 3.0 Task`:

```yaml
service: ai_task.generate_image
data:
  task_name: "Kitten"
  instructions: "A cute cat sitting on a windowsill, watercolor"
  entity_id: ai_task.alice_ai_image_art_3_0_task  # default
```

The image is stored in the [AI task media source](https://www.home-assistant.io/integrations/ai_task/) (`media-source://ai_task/image/...`); the service returns `media_source_id` and an `url` (valid for 1 hour). Output format is JPEG, the prompt is text only. The service account needs the `ai.imageGeneration.user` role.

## Thanks

Found the integration useful? Want to say thanks? A coffee for the author of the original integration is your way of thanking them. <kbd>[☕ Buy a coffee](https://mansmarthome.info/donate/?utm_source=github&utm_medium=referral&utm_campaign=gpt#%D1%81%D0%B8%D1%81%D1%82%D0%B5%D0%BC%D0%B0-%D0%B1%D1%8B%D1%81%D1%82%D1%80%D1%8B%D1%85-%D0%BF%D0%BB%D0%B0%D1%82%D0%B5%D0%B6%D0%B5%D0%B9)</kbd> I don't have accounts in any donation services, so you can thank me simply by sharing a link to this integration with friends and acquaintances. I would be glad to know that someone is using the fruits of my work.

A big thank you to everyone who supports the authors of both integrations:

![Thanks][donors-list]

## Notice

This is an independent community-developed fork. This integration is not affiliated with Yandex or Yandex Cloud in any way. Yandex AI Studio is a service provided by Yandex Cloud.

This integration is not an official Yandex product and is not supported by Yandex.

[donors-list]: https://github.com/user-attachments/assets/880000a7-0dc9-4bd3-99f5-bcf81eed1f8d
