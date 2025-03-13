import numpy as np
import matplotlib.pyplot as plt

from typing import List

def generate_signal(freq: List[float], amp: List[float], n: int):
    t = np.linspace(-np.pi, np.pi, n)
    signal = np.zeros(n, dtype=np.complex128)
    for f, a in zip(freq, amp):
        signal += a * np.sin(2 * np.pi * f * t) + a *  np.cos(2 * np.pi * f * t) * 1j
    return signal

def fft_process(signal):
    return np.fft.fft(signal)

def write_complex_to_file(data, filename):
    with open(filename, 'w') as f:
        for d in data:
            f.write(f'{d.real} {d.imag}\n')

def read_complex_from_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
    complex_data = []
    for line in data:
        real, imag = map(float, line.strip().split())
        complex_data.append(complex(real, imag))
    return complex_data

def plot_signal_and_fft(signal):
    fft_signal = fft_process(signal)
    fft_signal = np.abs(np.fft.fftshift(fft_signal)) ** 2

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.real(signal))
    plt.plot(np.imag(signal))
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(fft_signal)
    plt.grid(True)
    plt.show()

def plot_fft(fft_signal):
    plt.figure(figsize=(10, 6))
    plt.plot(np.abs(np.fft.fftshift(fft_signal)) ** 2)
    plt.grid(True)
    plt.show()

def main():
    freq = [1, 2.5,  3 * np.pi]
    amp = [1, 3.4, 2.1]
    signal = generate_signal(freq, amp, 1024)
    write_complex_to_file(signal, 'input.txt')
    plot_signal_and_fft(signal)

if __name__ == '__main__':
    fft_signal = read_complex_from_file('output.txt')
    plot_fft(fft_signal)