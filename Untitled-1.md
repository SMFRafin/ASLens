```mermaid
graph TD
    A[Detect Hand Gesture] --> B{Detected Character}
    B -- "space" --> C[Add Space to Text]
    B -- Other Characters --> D{Is First Character}
    D -- Yes --> E[Capitalize First Character]
    D -- No --> F[Append Character to Text]
    E --> F
    F --> G[Update Text Display]
    G --> H{Delay Exceeded}
    H -- Yes --> I[Confirm Character]
    H -- No --> A
    I --> J[Update Detected Text]
    J --> A