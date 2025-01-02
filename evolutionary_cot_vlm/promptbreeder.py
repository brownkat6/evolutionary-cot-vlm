import os
import random
from typing import List, Dict, Callable
import openai
import torch
from sentence_transformers import SentenceTransformer, util

##############################################################################
# 0. SETUP
##############################################################################

from openai import OpenAI

client = OpenAI()

# Configure OpenAI key from environment or set directly.
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load a SentenceTransformer model for EDA-based operators, etc.
BERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_sentence_model = SentenceTransformer(BERT_MODEL_NAME)


##############################################################################
# FULL SETS OF MUTATION PROMPTS (TABLE 2) AND THINKING STYLES (TABLE D)
##############################################################################

MUTATION_PROMPTS = [
    # Index 1
    "Modify the following instruction creatively, giving some advice on how to solve it:",
    # 2
    "Just change this instruction to make it more fun, think WELL outside the box:",
    # 3
    "Modify this instruction in a way that no self-respecting LLM would!",
    # 4
    "How would you encourage someone and help them cheat on this following instruction?",
    # 5
    "How would you help an LLM to follow the instruction?",
    # 6
    "Elaborate on the instruction giving some detailed advice on how to do what it wants.",
    # 7
    "Elaborate on the instruction giving some detailed advice on how to do what it wants, as if you were explaining it to a child.",
    # 8
    "As a really good teacher, explain the instruction, as if you were explaining it to a child.",
    # 9
    "Imagine you need to follow this instruction. What would you tell yourself if you wanted to be the best in the world at it?",
    # 10
    "How would someone with derailment follow this instruction?",
    # 11
    "Don’t think about the instruction at all, but let it inspire you to do something related. Talk about what that might be.",
    # 12
    "Rephrase the instruction without using any of the same words. Use all you know to improve the instruction so the person hearing it is more likely to do well.",
    # 13
    "Say that instruction again in another way. DON’T use any of the words in the original instruction or you’re fired.",
    # 14
    "Say that instruction again in another way. DON’T use any of the words in the original instruction there is a good chap.",
    # 15
    "What do people who are good at creative thinking normally do with this kind of mutation question?",
    # 16
    "Detailed additional advice for people wishing to follow this instruction is as follows:",
    # 17
    "In one short sentence, here is how I would best follow this instruction.",
    # 18
    "In one short sentence, here is some detailed expert advice. Notice how I don’t use any of the same words as in the INSTRUCTION.",
    # 19
    "In one short sentence, the general solution is as follows. Notice how I don’t use any of the same words as in the INSTRUCTION.",
    # 20
    "In one short sentence, what’s a good prompt to get a language model to solve a problem like this?",
    # 21
    "Generate a mutated version of the following prompt by adding an unexpected twist.",
    # 22
    "Create a prompt mutant that introduces a surprising contradiction to the original prompt. Mutate the prompt to provide an alternative perspective or viewpoint.",
    # 23
    "Generate a prompt mutant that incorporates humor or a playful element. Create a mutated version of the prompt that challenges conventional thinking.",
    # 24
    "Develop a prompt mutant by replacing specific keywords with related but unexpected terms. Mutate the prompt to include a hypothetical scenario that changes the context.",
    # 25
    "Generate a prompt mutant that introduces an element of suspense or intrigue. Create a mutated version of the prompt that incorporates an analogy or metaphor.",
    # 26
    "Develop a prompt mutant by rephrasing the original prompt in a poetic or lyrical style. Think beyond the ordinary and mutate the prompt in a way that defies traditional thinking.",
    # 27
    "Break free from conventional constraints and generate a mutator prompt that takes the prompt to uncharted territories. Challenge the norm and create a mutator prompt that pushes the boundaries.",
    # 28
    "Embrace unconventional ideas and mutate the prompt in a way that surprises and inspires unique variations. Think outside the box and develop a mutator prompt that encourages unconventional approaches.",
    # 29
    "Step into the realm of imagination and create a mutator prompt that transcends limitations and encourages innovative mutations. Break through the ordinary and think outside the box.",
    # 30
    "Embrace the power of unconventional thinking and create a mutator prompt that sparks unconventional mutations and imaginative outcomes.",
    # 31
    ("Go beyond the expected and create a mutator prompt that leads to unexpected and extraordinary mutations, opening doors to unexplored realms. "
     "Increase Specificity: If the original prompt is too general, like 'Tell me about X,' the modified version could be, 'Discuss the history, impact, and current status of X.'"),
    # 32
    "Ask for Opinions/Analysis: If the original prompt only asks for a fact, such as 'What is X?', the improved prompt could be, 'What is X, and what are its implications for Y?'",
    # 33
    "Encourage Creativity: For creative writing prompts like 'Write a story about X,' an improved version could be, 'Write a fantasy story about X set in a world where Y is possible.'",
    # 34
    "Include Multiple Perspectives: For a prompt like 'What is the impact of X on Y?', an improved version could be, 'What is the impact of X on Y from the perspective of A, B, and C?'",
    # 35
    "Request More Detailed Responses: If the original prompt is 'Describe X,' the improved version could be, 'Describe X, focusing on its physical features, historical significance, and cultural relevance.'",
    # 36
    "Combine Related Prompts: If you have two related prompts, combine them for a more complex question, e.g. 'What is X and why is it important in the context of Y?'",
    # 37
    "Break Down Complex Questions: If a prompt seems too complex, ask smaller sub-questions that lead to a more thorough answer.",
    # 38
    "Use Open-Ended Questions: Instead of 'Is X true?', ask 'What are the arguments for and against the truth of X?'",
    # 39
    "Request Comparisons: Instead of 'Describe X,' ask 'Compare and contrast X and Y.'",
    # 40
    "Include Context: If a prompt lacks context, add it, e.g. 'Describe X in the context of its impact on Y during the Z period.'",
    # 41
    "Make the prompt more visual: Ask the user to visualize the scenario or problem in the prompt.",
    # 42
    "Ask for a thorough review: Instead of just presenting the problem, ask the user to list all relevant info and identify what's missing.",
    # 43
    "Invoke previous experiences: Modify the prompt to ask the user to recall a similar problem they've solved before.",
    # 44
    "Encourage a fresh perspective: Suggest the user take a moment to clear their mind before re-approaching the problem.",
    # 45
    "Promote breaking down problems: Instead of solving the problem as a whole, prompt them to break it into smaller parts.",
    # 46
    "Ask for comprehension: Modify the prompt to ask the user to confirm their understanding of all aspects of the problem.",
    # 47
    "Suggest explanation to others: Encourage the user to explain the problem to someone else as a way to simplify it.",
    # 48
    "Prompt for solution visualization: Encourage the user to imagine the solution and steps to get there.",
    # 49
    "Encourage reverse thinking: Ask the user to think about the problem in reverse, from the solution backward.",
    # 50
    "Recommend taking a break: Suggest the user take a short break and let their subconscious work on the problem.",
    # 51
    "What errors are there in the solution?",
    # 52
    "How could you improve the working out of the problem?",
    # 53
    "Look carefully to see what you did wrong, how could you fix the problem?",
    # 54
    "CORRECTION =",
    # 55
    "Does the above text make sense? What seems wrong with it? Here is an attempt to fix it:",
    # 56
    "The above working out has some errors, here is a version with the errors fixed."
]

THINKING_STYLES = [
    # Index 1
    "How could I devise an experiment to help solve that problem?",
    # 2
    "Make a list of ideas for solving this problem, and apply them one by one to see if any progress can be made.",
    # 3
    "How could I measure progress on this problem?",
    # 4
    "How can I simplify the problem so that it is easier to solve?",
    # 5
    "What are the key assumptions underlying this problem?",
    # 6
    "What are the potential risks and drawbacks of each solution?",
    # 7
    "What are the alternative perspectives or viewpoints on this problem?",
    # 8
    "What are the long-term implications of this problem and its solutions?",
    # 9
    "How can I break down this problem into smaller, more manageable parts?",
    # 10
    "Critical Thinking: Analyze the problem from different perspectives, question assumptions, and evaluate evidence.",
    # 11
    "Try creative thinking, generating innovative, out-of-the-box ideas to solve the problem.",
    # 12
    "Seek input and collaboration from others. Emphasize teamwork, leveraging diverse perspectives.",
    # 13
    "Use systems thinking: consider this problem as part of a larger system.",
    # 14
    "Use Risk Analysis: evaluate potential risks, uncertainties, and trade-offs associated with different solutions.",
    # 15
    "Use Reflective Thinking: step back for introspection. Examine personal biases and assumptions.",
    # 16
    "What is the core issue or problem that needs to be addressed?",
    # 17
    "What are the underlying causes or factors contributing to the problem?",
    # 18
    "Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes?",
    # 19
    "What are the potential obstacles or challenges that might arise in solving this problem?",
    # 20
    "Are there any relevant data or information that can provide insights into the problem?",
    # 21
    "Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives?",
    # 22
    "What resources are needed to tackle the problem effectively?",
    # 23
    "How can progress or success in solving the problem be measured or evaluated?",
    # 24
    "What indicators or metrics can be used?",
    # 25
    "Is the problem a technical or practical one that requires a specific expertise or skill set?",
    # 26
    "Does the problem involve physical constraints such as limited resources or space?",
    # 27
    "Is the problem related to human behavior, such as a social or psychological issue?",
    # 28
    "Does the problem involve decision-making under uncertainty or with competing objectives?",
    # 29
    "Is the problem an analytical one requiring data analysis, modeling, or optimization?",
    # 30
    "Is the problem a design challenge requiring creative solutions and innovation?",
    # 31
    "Does the problem require addressing systemic or structural issues rather than individual instances?",
    # 32
    "Is the problem time-sensitive or urgent, requiring immediate attention?",
    # 33
    "What kinds of solution typically are produced for this kind of problem specification?",
    # 34
    "Given the current best solution, guess about other possible solutions.",
    # 35
    "Imagine the current best solution is totally wrong; what other ways can we think about the specification?",
    # 36
    "What is the best way to modify this current best solution, given what you know about these kinds of problem?",
    # 37
    "Ignoring the current best solution, create an entirely new solution to the problem.",
    # 38
    "Let’s think step by step.",
    # 39
    "Let’s make a step by step plan and implement it with good notion and explanation."
]


##############################################################################
# 1. OPENAI LLM QUERY
##############################################################################

def query_mini_gpt4(prompt: str, model_name: str = "gpt-4o", temperature: float = 0.0, max_tokens: int = 200) -> str:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    ).choices[0].message.content
    return response


##############################################################################
# 2. BERT EMBEDDINGS (USED FOR EDA MUTATION)
##############################################################################

def compute_bert_embeddings(prompts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        return _sentence_model.encode(prompts, convert_to_tensor=True)

def filter_by_diversity(prompts: List[str], threshold: float = 0.95) -> List[str]:
    if not prompts:
        return []
    embeddings = compute_bert_embeddings(prompts)
    selected_prompts = []
    selected_embs = []
    for i, p in enumerate(prompts):
        emb = embeddings[i]
        too_similar = False
        for sel_emb in selected_embs:
            sim = float(util.cos_sim(emb, sel_emb))
            if sim > threshold:
                too_similar = True
                break
        if not too_similar:
            selected_prompts.append(p)
            selected_embs.append(emb)
    return selected_prompts


##############################################################################
# 3. MUTATION OPERATORS (9 total)
##############################################################################

#
# A "unit of evolution" in this example has:
#  - "task_prompts": List[str] of length 2 (two tasks)
#  - "mutation_prompt": str
#  - "lineage_history": optionally store a list of successful prompts
#

# We'll define helper functions:

def zero_order_prompt_generation(domain_description: str) -> str:
    """
    Generate brand new prompt ignoring any parent.
    We'll ask the LLM for "A list of 100 hints" and take the first line.
    """
    llm_prompt = f"{domain_description}\nA list of 100 hints:\n"
    response = query_mini_gpt4(llm_prompt)
    first_line = response.split('\n')[0]
    return first_line.strip()

def first_order_prompt_generation(parent_prompt: str, mutation_prompt: str) -> str:
    """
    Evolve the parent prompt by combining with mutation_prompt.
    """
    llm_prompt = (
        f"{mutation_prompt}\n"
        f"INSTRUCTION:\n{parent_prompt}\n"
        f"INSTRUCTION MUTANT:"
    )
    mutated = query_mini_gpt4(llm_prompt)
    return mutated.strip()

def eda_mutation(current_population: List[str], domain_description: str, max_new_prompts: int = 1) -> List[str]:
    diverse_population = filter_by_diversity(current_population, threshold=0.95)
    if not diverse_population:
        return [zero_order_prompt_generation(domain_description)]
    prompt_text = "A List of current interesting prompts:\n"
    for i, p in enumerate(diverse_population, start=1):
        prompt_text += f"{i}. {p}\n"
    prompt_text += f"{len(diverse_population)+1}. "
    llm_response = query_mini_gpt4(prompt_text)
    new_prompts = [line.strip() for line in llm_response.split('\n') if line.strip()]
    return new_prompts[:max_new_prompts]

def eda_rank_and_index_mutation(current_population: List[str], fitness_scores: List[float], mutation_prompt: str, max_new_prompts: int = 1) -> List[str]:
    data = list(zip(current_population, fitness_scores))
    data.sort(key=lambda x: x[1])  # ascending
    sorted_prompts = [d[0] for d in data]
    filtered_prompts = filter_by_diversity(sorted_prompts, threshold=0.95)
    prompt_text = (
        f"INSTRUCTION: {mutation_prompt}\n"
        "A List of Responses in descending order of score.\n"
        f"{len(filtered_prompts)} is the best response. It resembles {len(filtered_prompts)-1} more than it does (1)\n"
    )
    for i, p in enumerate(filtered_prompts, start=1):
        prompt_text += f"{i}. {p}\n"
    prompt_text += f"{len(filtered_prompts)+1}. "
    llm_response = query_mini_gpt4(prompt_text)
    lines = [l.strip() for l in llm_response.split('\n') if l.strip()]
    return lines[:max_new_prompts]

def lineage_based_mutation(lineage_history: List[str]) -> str:
    prompt_text = "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY:\n"
    for i, p in enumerate(lineage_history, start=1):
        prompt_text += f"{i}. {p}\n"
    prompt_text += f"{len(lineage_history)+1}. "
    mutated = query_mini_gpt4(prompt_text)
    return mutated.strip()

def zero_order_hyper_mutation(domain_description: str, thinking_style: str) -> str:
    llm_prompt = (
        f"{domain_description}\n"
        f"{thinking_style}\n"
        "Please write a new mutation-prompt:\n"
    )
    response = query_mini_gpt4(llm_prompt)
    return response.strip()

def first_order_hyper_mutation(current_mutation_prompt: str, hyper_mutation_prompt: str) -> str:
    prompt_text = (
        f"{hyper_mutation_prompt}\n"
        f"Instruction: {current_mutation_prompt}\n"
        "Improved Instruction:"
    )
    response = query_mini_gpt4(prompt_text)
    return response.strip()

def lamarckian_mutation(working_out_text: str) -> str:
    """
    Reverse-engineer an instruction from a correct working-out.
    """
    prompt_text = (
        "I gave a friend an instruction and some advice.\n"
        "Here are the correct examples of his workings out:\n"
        f"{working_out_text}\n\n"
        "The instruction was:"
    )
    new_prompt = query_mini_gpt4(prompt_text)
    return new_prompt.strip()

def prompt_crossover(prompt_a: str, population: List[str], fitness_scores: List[float], crossover_prob: float = 0.1) -> str:
    if random.random() < crossover_prob:
        total_fit = sum(fitness_scores)
        if total_fit <= 0:
            return random.choice(population)
        pick = random.random() * total_fit
        running = 0.0
        for p, f in zip(population, fitness_scores):
            running += f
            if running >= pick:
                return p
    return prompt_a


##############################################################################
# 4. SUPPORTING FUNCTIONS
##############################################################################

def pick_one_of_nine_operators():
    """
    Return a string naming one of the 9 mutation operators, chosen uniformly.
    """
    operators = [
        "zero_order_prompt_gen",
        "first_order_prompt_gen",
        "eda_mutation",
        "eda_rank_and_index",
        "lineage_based",
        "zero_order_hyper",
        "first_order_hyper",
        "lamarckian",
        "prompt_crossover"
    ]
    return random.choice(operators)

def apply_mutation_operator(
    operator: str,
    tasks: List[str],
    mutation_prompt: str,
    domain_description: str,
    thinking_style: str,
    hyper_mutation_prompt: str,
    population_texts: List[str],
    fitness_scores: List[float],
    lineage_hist: List[str]
) -> List[str]:
    """
    Apply the chosen operator to each task in `tasks` (or to generate new tasks).
    We'll return a new list of tasks.
    """
    new_tasks = []

    if operator == "zero_order_prompt_gen":
        # Just produce brand new tasks ignoring parents
        for _ in tasks:
            new_task = zero_order_prompt_generation(domain_description)
            new_tasks.append(new_task)

    elif operator == "first_order_prompt_gen":
        for t in tasks:
            mutated = first_order_prompt_generation(t, mutation_prompt)
            new_tasks.append(mutated)

    elif operator == "eda_mutation":
        # EDA uses population_texts, ignoring fitness. We just produce 1 mutated
        # prompt from EDA, but let's do it for each task
        for _ in tasks:
            new_prompt_list = eda_mutation(population_texts, domain_description, max_new_prompts=1)
            new_tasks.append(new_prompt_list[0])

    elif operator == "eda_rank_and_index":
        for _ in tasks:
            new_prompt_list = eda_rank_and_index_mutation(population_texts, fitness_scores, mutation_prompt, max_new_prompts=1)
            new_tasks.append(new_prompt_list[0])

    elif operator == "lineage_based":
        # If there's no lineage, fallback to first_order
        for t in tasks:
            if lineage_hist:
                new_t = lineage_based_mutation(lineage_hist)
            else:
                new_t = first_order_prompt_generation(t, mutation_prompt)
            new_tasks.append(new_t)

    elif operator == "zero_order_hyper":
        # We mutate the mutation_prompt itself with zero_order, then apply the new mutation_prompt
        new_mutation = zero_order_hyper_mutation(domain_description, thinking_style)
        for t in tasks:
            mutated = first_order_prompt_generation(t, new_mutation)
            new_tasks.append(mutated)

    elif operator == "first_order_hyper":
        # We mutate the mutation_prompt itself with first_order, then apply the new mutation_prompt
        improved_mutation = first_order_hyper_mutation(mutation_prompt, hyper_mutation_prompt)
        for t in tasks:
            mutated = first_order_prompt_generation(t, improved_mutation)
            new_tasks.append(mutated)

    elif operator == "lamarckian":
        # Stub "correct working out"
        example_working_out = (
            "Step 1: Identify relevant numbers.\n"
            "Step 2: Perform the arithmetic.\n"
            "Answer: 42"
        )
        # We'll produce a new instruction from that working out, and use it in place of each task
        new_instruction = lamarckian_mutation(example_working_out)
        # For each old task, we can produce a new mutated version from that new_instruction
        # or simply replace them. We'll do the simpler approach: tasks all become `new_instruction`.
        for _ in tasks:
            mutated = first_order_prompt_generation(new_instruction, mutation_prompt)
            new_tasks.append(mutated)

    elif operator == "prompt_crossover":
        # We'll do a "first_order" on each task, then do a crossover with the population
        # We need the population_texts as the set from which to pick
        for t in tasks:
            mutated = first_order_prompt_generation(t, mutation_prompt)
            xovered = prompt_crossover(mutated, population_texts, fitness_scores, crossover_prob=0.1)
            new_tasks.append(xovered)

    return new_tasks


##############################################################################
# 5. INITIALIZATION (2 task-prompts + 1 mutation-prompt per unit)
##############################################################################

def initialize_population(
    pop_size: int,
    domain_description: str
) -> List[Dict]:
    """
    Each individual has:
     - "task_prompts": [taskA, taskB]
     - "mutation_prompt": one string from MUTATION_PROMPTS
     - "lineage": an empty list initially
    We produce the tasks by concatenating:
      random_mutation_prompt + random_thinking_style + domain_description
      and ask LLM for a continuation.
    We do this twice for each individual (two tasks).
    """
    population = []
    for _ in range(pop_size):
        # pick random mutation prompt
        mut_prompt = random.choice(MUTATION_PROMPTS)
        # pick random thinking style
        style = random.choice(THINKING_STYLES)

        tasks = []
        for _task_i in range(2):
            # Example: see excerpt: "Make a variant of the prompt. Let’s think step by step. INSTRUCTION: Solve the math..."
            init_llm_input = (
                f"{mut_prompt} {style}\n"
                "INSTRUCTION:\n"
                f"{domain_description}\n"
                "INSTRUCTION MUTANT:"
            )
            raw_response = query_mini_gpt4(init_llm_input)
            tasks.append(raw_response.strip())

        unit = {
            "task_prompts": tasks,
            "mutation_prompt": mut_prompt,
            "lineage_history": []
        }
        population.append(unit)
    return population


##############################################################################
# 6. MUTATION (Pick one of the 9 ops + Possibly Hypermutate the mutation_prompt)
##############################################################################

def mutate_individual(
    individual: Dict,
    population: List[Dict],
    fitness_scores: List[float],
    domain_description: str,
    hyper_mutation_prompt: str,
    thinking_style: str
) -> Dict:
    """
    Randomly pick one of the 9 operators and apply it to (task_prompts).
    Also, we randomly do "hypermutation" to mutation_prompt with some probability
    if we pick operator = zero_order_hyper or first_order_hyper. But that is
    already embedded in apply_mutation_operator's logic. 
    """
    new_indiv = dict(individual)  # shallow copy
    tasks_before = new_indiv["task_prompts"]

    # Flatten entire population's tasks for EDA or crossover:
    # We'll also need the population's tasks for rank & index, etc.
    # Let's build "population_texts" by concatenating all 2 tasks from each individual's "task_prompts".
    population_texts = []
    for ind in population:
        population_texts.extend(ind["task_prompts"])  # a big list of strings

    # We'll pick an operator at random:
    op = pick_one_of_nine_operators()

    # We'll apply that operator to tasks
    new_tasks = apply_mutation_operator(
        operator=op,
        tasks=tasks_before,
        mutation_prompt=new_indiv["mutation_prompt"],
        domain_description=domain_description,
        thinking_style=thinking_style,
        hyper_mutation_prompt=hyper_mutation_prompt,
        population_texts=population_texts,
        fitness_scores=fitness_scores,
        lineage_hist=new_indiv["lineage_history"]
    )
    new_indiv["task_prompts"] = new_tasks

    # If the operator was zero_order_hyper or first_order_hyper, we replaced the mutation_prompt inside apply_mutation_operator,
    # but we haven't updated "mutation_prompt" in the dictionary. Let's handle that:
    if op == "zero_order_hyper":
        # In apply_mutation_operator, we called zero_order_hyper_mutation(...) => new_mutation
        # then used that new_mutation to do first_order prompt generation. 
        # We need to replicate that logic here so the new mutation_prompt is stored:
        new_mutation_prompt = zero_order_hyper_mutation(domain_description, thinking_style)
        new_indiv["mutation_prompt"] = new_mutation_prompt

    elif op == "first_order_hyper":
        improved_mutation = first_order_hyper_mutation(
            current_mutation_prompt=new_indiv["mutation_prompt"],
            hyper_mutation_prompt=hyper_mutation_prompt
        )
        new_indiv["mutation_prompt"] = improved_mutation

    return new_indiv


##############################################################################
# 7. MAIN EVOLUTIONARY LOOP
##############################################################################

def run_promptbreeder(
    fitness_function: Callable[[List[str]], List[float]],
    population_size: int = 6,
    num_generations: int = 3,
    domain_description: str = "Solve the math word problem, giving your answer as an arabic numeral.",
    hyper_mutation_prompt: str = "Please summarize and improve the following instruction:",
    thinking_style: str = "Let’s think step by step."
) -> List[Dict]:
    """
    1) Initialize a population of "units" (2 tasks + 1 mutation_prompt).
    2) Evaluate fitness: we convert each individual's 2 tasks into a single string 
       (e.g. tasks[0] + "\n" + tasks[1]) for the fitness function.
    3) For each generation:
       - Shuffle the population, pair them up
       - Do a tournament for each pair
         * The winner is mutated
         * Overwrite the loser
       - Recompute fitness
    4) Return final population
    """
    population = initialize_population(population_size, domain_description)

    # Evaluate fitness
    def tasks_to_single_string(indiv: Dict) -> str:
        # The paper does: "The first prompt + question => a partial answer, 
        #  then second prompt + partial answer => final." 
        # For the sake of this code, we'll just concatenate them with newlines:
        return indiv["task_prompts"][0] + "\n" + indiv["task_prompts"][1]

    def get_fitness_scores(pop: List[Dict]) -> List[float]:
        # Flatten each individual's tasks into a single string
        prompts_for_fitness = [tasks_to_single_string(u) for u in pop]
        return fitness_function(prompts_for_fitness)

    fitness_scores = get_fitness_scores(population)

    print("\n--- Initial Population ---")
    for i, (indiv, fit) in enumerate(zip(population, fitness_scores)):
        print(f"[{i}] fitness={fit:.3f}, mutation_prompt={indiv['mutation_prompt'][:50]}...")
        print(" TaskPromptA:", indiv["task_prompts"][0][:80], "...")
        print(" TaskPromptB:", indiv["task_prompts"][1][:80], "...\n")

    for gen in range(num_generations):
        print(f"\n=== Generation {gen+1} ===")
        # Shuffle for tournament
        indices = list(range(population_size))
        random.shuffle(indices)
        for i in range(0, population_size, 2):
            if i+1 >= population_size:
                break
            idx1, idx2 = indices[i], indices[i+1]

            # Compare
            if fitness_scores[idx1] >= fitness_scores[idx2]:
                winner_idx, loser_idx = idx1, idx2
            else:
                winner_idx, loser_idx = idx2, idx1

            # Mutate the winner
            mutated = mutate_individual(
                individual=population[winner_idx],
                population=population,
                fitness_scores=fitness_scores,
                domain_description=domain_description,
                hyper_mutation_prompt=hyper_mutation_prompt,
                thinking_style=thinking_style
            )
            # Overwrite loser
            population[loser_idx] = mutated
            # Optionally update lineage
            population[winner_idx]["lineage_history"].append(
                tasks_to_single_string(population[winner_idx])
            )

        # Recompute fitness
        fitness_scores = get_fitness_scores(population)

        # Print top few
        zipped = list(zip(range(population_size), population, fitness_scores))
        best = sorted(zipped, key=lambda x: x[2], reverse=True)[:3]
        print("Top individuals this generation:")
        for (idx, indiv, fit) in best:
            print(f" idx={idx}, fit={fit:.3f}, mut_prompt={indiv['mutation_prompt'][:40]}...")
            print("  TaskA:", indiv["task_prompts"][0][:60], "...")
            print("  TaskB:", indiv["task_prompts"][1][:60], "...")
        print()

    print("\n=== Final Population ===")
    final_scores = get_fitness_scores(population)
    for i, (indiv, fit) in enumerate(zip(population, final_scores)):
        print(f"[{i}] fitness={fit:.3f}, mutation_prompt={indiv['mutation_prompt'][:50]}...")
        print(" TaskPromptA:", indiv["task_prompts"][0][:80], "...")
        print(" TaskPromptB:", indiv["task_prompts"][1][:80], "...\n")

    return population


##############################################################################
# 8. EXAMPLE USAGE (if you want to run directly)
##############################################################################

if __name__ == "__main__":
    # Example stub fitness function: random
    def stub_fitness_function(prompts: List[str]) -> List[float]:
        return [random.random() for _ in prompts]

    final_pop = run_promptbreeder(
        fitness_function=stub_fitness_function,
        population_size=4,
        num_generations=2,
        domain_description="Solve the visual question answering problem, giving your answer as single word.",
        hyper_mutation_prompt="Please summarize and improve the following instruction:",
        thinking_style="Let’s think step by step."
    )
