flowchart TB


   %% ==========================
   %%         NODES
   %% ==========================


   A["User Input"]
   B["Sentiment Agent"]
   C["Context Agent"]
   R["Routing Agent"]
   DEC{"Select Database(s)?"}


   VDB[("Voting DB")]
   BDB[("Bio DB")]
   SDB[("Social DB")]
   PDB[("Policy DB")]


   D{"Is Data Found?"}
   E["Tone Agent"]
   PDB2[("Persona DB")]
   F["Deflection Agent"]
   G["Response Composer"]
   CMD[("Chat Memory DB")]
   H["Fact Checking Agent"]
   FK(("Factual Knowledge Base"))
   I["Final Output"]


   %% ==========================
   %%       FLOWS & LABELS
   %% ==========================


   %% 1. Parallel flow: User Input -> Sentiment & Context
   A -- "user text" --> B
   A -- "user text" --> C


   %% 2. Context -> Routing
   C -- "extracted context" --> R
   R -- "route by context" --> DEC


   %% 3. Decide which DB(s) to query
   DEC -- "Voting?" --> VDB
   VDB -- "voting data" --> R


   DEC -- "Bio?" --> BDB
   BDB -- "bio data" --> R


   DEC -- "Social?" --> SDB
   SDB -- "tweets/interviews" --> R


   DEC -- "Policy?" --> PDB
   PDB -- "policy docs" --> R


   %% 4. Aggregated data -> Is Data Found?
   R --> D


   %% 5. Data found => Tone; Not found => Deflect
   D -- "Yes
   (pass aggregated data)" --> E
   D -- "No
   (pass context)" --> F


   %% 6. Sentiment to Tone or Deflection
   B -- "sentiment" --> E
   B -- "sentiment" --> F


   %% 7. Tone Agent checks Persona DB
   E -- "request style" --> PDB2
   PDB2 -- "persona style" --> E


   %% Tone Agent -> "tone + aggregated data"
   E -- "tone + aggregated data" --> G


   %% DeflAgent -> "deflection + context"
   F -- "deflection + context" --> G


   %% 8. Response Composer consults Chat Memory
   G -- "need chat history" --> CMD
   CMD -- "chat memory" --> G


   %% 9. Fact Check
   G -- "draft response" --> H
   H -- "verify with facts" --> FK
   FK -- "factual refs" --> H
   H -- "verified response" --> I


   %% ==========================
   %%     STYLING (purple)
   %% ==========================
   style A fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style B fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style C fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style R fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style DEC fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000


   style VDB fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style BDB fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style SDB fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style PDB fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000


   style D fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style E fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style PDB2 fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style F fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style G fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style CMD fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000


   style H fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
   style FK fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000


   style I fill:#E6CCFF,stroke:#7B3F7B,stroke-width:2px,color:#000000
