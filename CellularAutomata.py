# Code based on NotCSV channel

import numpy as np
import pygame
import time

pygame.init()
# Window size
width, height = 900, 900
screen = pygame.display.set_mode((height, width))

# BG color
bg = 25, 25, 25
screen.fill(bg)

# Cell number
nxC, nyC = 450, 450

# Pixels by cell
dimCW = int(width / nxC)
dimCH = int(height / nyC)

# Create empty grid
gameState = np.zeros((nxC, nyC))
for y in range(0, nyC):
    for x in range(0, nxC):
        poly = [(x * dimCW, y * dimCH),
                ((x + 1) * dimCW, y * dimCH),
                ((x + 1) * dimCW, (y + 1) * dimCH),
                (x * dimCW, (y + 1) * dimCH)]
        pygame.draw.polygon(screen, (128, 128, 128), poly, 1)

# Create first alive cell in the middle of the top row
gameState[int(nxC / 2), 0] = 1
pauseExec = False
y = 0

# We select rule 90 (from 0 to 255)
rules = list(np.binary_repr(90, width=8))
rules.reverse()

# Generation loop
while y < nyC:
    # Copy of the game configure depending on game-1
    newGameState = np.copy(gameState)
    # time.sleep(0.01)

    # Every row is a generation
    for x in range(0, nxC):
        # Apply the rule and modify the environment
        rule_index = 4 * gameState[(x - 1) % nxC, y] + 2 * gameState[x, y] + 1 * gameState[(x + 1) % nxC, y]
        newGameState[x, (y + 1) % nyC] = rules[int(rule_index)]
        poly = [(x * dimCW, y * dimCH),
                ((x + 1) * dimCW, y * dimCH),
                ((x + 1) * dimCW, (y + 1) * dimCH),
                (x * dimCW, (y + 1) * dimCH)]

        if newGameState[x, y] == 1:
            pygame.draw.polygon(screen, (255, 255, 255), poly, 0)

    # We limit the y grow to the number of cells
    y += 1

    gameState = np.copy(newGameState)
    time.sleep(0.01)
    pygame.display.flip()

while 1:
    time.sleep(10.0)
    print('End')
