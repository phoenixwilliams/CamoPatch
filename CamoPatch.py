import numpy as np
import os
import math
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def l2(adv_patch, orig_patch):
    assert adv_patch.shape == orig_patch.shape
    return np.sum((adv_patch - orig_patch) ** 2)


def sh_selection(n_queries, it):
    """ schedule to decrease the parameter p """

    t = max((float(n_queries - it) / n_queries - .0) ** 1., 0) * .75

    return t


def update_location(loc_new, h_i, h, s):
    loc_new += np.random.randint(low=-h_i, high=h_i + 1, size=(2,))
    loc_new = np.clip(loc_new, 0, h - s)
    return loc_new


def render(x, w):
    phenotype = np.ones((w, w, 3))
    radius_avg = (phenotype.shape[0] + phenotype.shape[1]) / 2 / 6
    for row in x:
        overlay = phenotype.copy()
        cv2.circle(
            overlay,
            center=(int(row[1] * w), int(row[0] * w)),
            radius=int(row[2] * radius_avg),
            color=(int(row[3] * 255), int(row[4] * 255), int(row[5] * 255)),
            thickness=-1,
        )
        alpha = row[6]
        phenotype = cv2.addWeighted(overlay, alpha, phenotype, 1 - alpha, 0)

    return phenotype/255.


def mutate(soln, mut):
    """Mutates specie for evolution.

    Args:
        specie (species.Specie): Specie to mutate.

    Returns:
        New Specie class, that has been mutated.
        :param soln:
    """
    new_specie = soln.copy()

    # Randomization for Evolution
    genes = soln.shape[0]
    length = soln.shape[1]
    y = np.random.randint(0, genes)
    change = np.random.randint(0, length + 1)

    if change >= length + 1:
        change -= 1
        i, j = y, np.random.randint(0, genes)
        i, j, s = (i, j, -1) if i < j else (j, i, 1)
        new_specie[i: j + 1] = np.roll(new_specie[i: j + 1], shift=s, axis=0)
        y = j

    selection = np.random.choice(length, size=change, replace=False)

    if np.random.rand() < mut:
        new_specie[y, selection] = np.random.rand(len(selection))
    else:
        new_specie[y, selection] += (np.random.rand(len(selection)) - 0.5) / 3
        new_specie[y, selection] = np.clip(new_specie[y, selection], 0, 1)

    return new_specie


class Attack:
    def __init__(self, params):
        self.params = params
        self.process = []

    def completion_procedure(self, adversarial, x_adv, queries, loc, patch, loss_function):
        data = {
            "orig": self.params["x"],
            "adversary": x_adv,
            "adversarial": adversarial,
            "queries": queries,
            "loc": loc,
            "patch": patch,
            "patch_width": int(math.ceil(self.params["eps"] ** .5)),
            "final_prediction": loss_function.get_label(x_adv),
            "process": self.process
        }

        np.save(self.params["save_directory"], data, allow_pickle=True)

    def optimise(self, loss_function):
        # initialize
        x = self.params["x"]
        c, h, w = self.params["c"], self.params["h"], self.params["w"]
        eps = self.params["eps"]
        s = int(math.ceil(eps ** .5))

        patch_geno = np.random.rand(self.params["N"], 7)
        patch = render(patch_geno, s)
        loc = np.random.randint(h - s, size=2)

        update_loc_period = self.params["update_loc_period"]

        x_adv = x.copy()
        x_adv[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch
        x_adv = np.clip(x_adv, 0., 1.)
        adversarial, loss = loss_function(x_adv)
        l2_curr = l2(adv_patch=patch, orig_patch=x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy())

        patch_counter = 0
        n_queries = self.params["n_queries"]
        for it in tqdm(range(1, n_queries)):
            patch_counter += 1
            if patch_counter < update_loc_period:
                patch_new_geno = mutate(patch_geno, self.params["mut"])
                patch_new = render(patch_new_geno, s)
                x_adv_new = x.copy()
                x_adv_new[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :] = patch_new
                x_adv_new = np.clip(x_adv_new, 0., 1.)
                # evaluate new solutions

                adversarial_new, loss_new = loss_function(x_adv_new)

                orig_patch = x[loc[0]: loc[0] + s, loc[1]: loc[1] + s, :].copy()
                l2_new = l2(adv_patch=patch_new, orig_patch=orig_patch)

                if adversarial == True and adversarial_new == True:

                    if l2_new < l2_curr:
                        loss = loss_new
                        adversarial = adversarial_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new

                else:
                    if loss_new < loss: # minimization
                        loss = loss_new
                        adversarial = adversarial_new
                        patch = patch_new
                        patch_geno = patch_new_geno
                        x_adv = x_adv_new
                        l2_curr = l2_new

            else:
                patch_counter = 0

                # location update
                sh_i = int(max(sh_selection(n_queries, it) * h, 0))
                loc_new = loc.copy()
                loc_new = update_location(loc_new, sh_i, h, s)
                x_adv_new = x.copy()
                x_adv_new[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :] = patch
                x_adv_new = np.clip(x_adv_new, 0., 1.)
                # evaluate new solution

                adversarial_new, loss_new = loss_function(x_adv_new)

                orig_patch_new = x[loc_new[0]: loc_new[0] + s, loc_new[1]: loc_new[1] + s, :].copy()
                l2_new = l2(adv_patch=patch, orig_patch=orig_patch_new)

                if adversarial == True and adversarial_new == True:
                    if l2_new < l2_curr:
                        loss = loss_new
                        adversarial = adversarial_new
                        loc = loc_new

                        x_adv = x_adv_new
                        l2_curr = l2_new

                else:
                    diff = loss_new - loss
                    curr_temp = self.params["temp"] / (it +1)
                    metropolis = math.exp(-diff/curr_temp)

                    if loss_new < loss or np.random.rand() < metropolis:  # minimization # first check
                        loss = loss_new
                        adversarial = adversarial_new
                        loc = loc_new
                        x_adv = x_adv_new
                        l2_curr = l2_new

            self.process.append([loc, patch_geno])

        self.completion_procedure(adversarial, x_adv, it, loc, patch, loss_function)
        return