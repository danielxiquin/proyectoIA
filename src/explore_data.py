import csv
from collections import Counter

def load_dataset(filepath):
    tickets = []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickets.append(row)
    return tickets

if __name__ == "__main__":
    tickets = load_dataset("data/customer_support_tickets.csv")
    
    print(f"Total de tickets: {len(tickets)}")
    print(f"\nColumnas disponibles:")
    print(list(tickets[0].keys()))
    
    categories = [t['Ticket Type'] for t in tickets]
    counter = Counter(categories)
    print(f"\nDistribución de categorías:")
    for cat, count in counter.most_common():
        print(f"  {cat}: {count} ({count/len(tickets)*100:.1f}%)")
    
    print(f"\nEjemplo de ticket:")
    print(f"  Texto: {tickets[0]['Ticket Description'][:100]}...")
    print(f"  Categoría: {tickets[0]['Ticket Type']}")