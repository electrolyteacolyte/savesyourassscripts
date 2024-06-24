import tkinter as tk
import math

class ScientificCalculator:
    def __init__(self, master):
        self.master = master
        master.title("Scientific Calculator")

        self.entry = tk.Entry(master, width=30, font=('Arial', 14))
        self.entry.grid(row=0, column=0, columnspan=6)

        buttons = [
            ('7', 1, 0), ('8', 1, 1), ('9', 1, 2), ('/', 1, 3),
            ('4', 2, 0), ('5', 2, 1), ('6', 2, 2), ('*', 2, 3),
            ('1', 3, 0), ('2', 3, 1), ('3', 3, 2), ('-', 3, 3),
            ('0', 4, 0), ('.', 4, 1), ('=', 4, 2), ('+', 4, 3),
            ('sin', 5, 0), ('cos', 5, 1), ('tan', 5, 2), ('sqrt', 5, 3),
            ('(', 6, 0), (')', 6, 1), ('AC', 6, 2), ('^', 6, 3),
            ('log', 7, 0), ('log2', 7, 1), ('log10', 7, 2), ('exp', 7, 3),
            ('pi', 8, 0), ('e', 8, 1)
        ]

        for (text, row, col) in buttons:
            button = tk.Button(master, text=text, width=5, height=2, font=('Arial', 14),
                               command=lambda t=text: self.click(t))
            button.grid(row=row, column=col)

    def click(self, key):
        if key == '=':
            try:
                result = eval(self.entry.get())
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'AC':
            self.entry.delete(0, tk.END)
        elif key == 'sqrt':
            try:
                result = math.sqrt(eval(self.entry.get()))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'sin':
            try:
                result = math.sin(math.radians(eval(self.entry.get())))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'cos':
            try:
                result = math.cos(math.radians(eval(self.entry.get())))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'tan':
            try:
                result = math.tan(math.radians(eval(self.entry.get())))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'log':
            try:
                result = math.log(eval(self.entry.get()))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'log2':
            try:
                result = math.log2(eval(self.entry.get()))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'log10':
            try:
                result = math.log10(eval(self.entry.get()))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'exp':
            try:
                result = math.exp(eval(self.entry.get()))
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, str(result))
            except:
                self.entry.delete(0, tk.END)
                self.entry.insert(tk.END, "Error")
        elif key == 'pi':
            self.entry.insert(tk.END, str(math.pi))
        elif key == 'e':
            self.entry.insert(tk.END, str(math.e))
        else:
            self.entry.insert(tk.END, key)

def main():
    root = tk.Tk()
    calculator = ScientificCalculator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
