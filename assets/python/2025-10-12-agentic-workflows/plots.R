# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %%
library(ggbeeswarm)
library(jsonlite)
library(patchwork)
library(tidyverse)

# %%
screening_to_str <- function(col) {
    ifelse(col, "Pass", "Fail")
}

priority_to_str <- function(col) {
    ifelse(is.na(col), "Screening Failed", str_to_title(col))
}

labels <- bind_rows(
    read_csv("train_articles.csv", col_types = "clc") %>%
        mutate(dataset = "Train"),
    read_csv("test_articles.csv", col_types = "clc") %>%
        mutate(dataset = "Test")
) %>%
    mutate(
        my_screening = screening_to_str(my_screening),
        my_priority = priority_to_str(my_priority),
        dataset = factor(dataset, levels = c("Train", "Test")),
        my_priority = factor(
            my_priority,
            levels = c("Screening Failed", "Low", "Medium", "High")
        )
    )

# %%
barplot <- function(df, title, subtitle, x_label, y_label="Fraction of Articles (%)", alpha="Dataset", legend.position="right") {

    if (length(unique(df$plotting_col)) == 4) {
        palette <- c("#6c7a89", "#006BA2", "#FFB300", "#DB444B")
    } else if (length(unique(df$plotting_col)) == 3) {
        palette <- c("#6c7a89", "#006BA2", "#DB444B")
    } else if (length(unique(df$plotting_col)) == 2) {
        palette <- c("#DB444B", "#006BA2")
    } else {
        stop("Too many unique values in plotting_col for the defined palette.")
    }

    df %>%
        ggplot(aes(x = plotting_col, y = prop, fill = plotting_col, alpha = dataset)) +
            geom_col(position = position_dodge()) +
            labs(
                title = title,
                subtitle = subtitle,
                x = x_label,
                y = y_label,
                alpha = alpha
            ) +
        scale_fill_manual(values = palette, guide = "none") +
        scale_alpha_manual(values = c("Train" = 1.0, "Test" = 0.7)) +
        scale_y_continuous(expand = expansion(mult = c(0, 0.05)), limits = c(0, NA)) +
        theme_minimal() +
        theme(
            plot.title = element_text(size = 16, face = "bold", hjust = 0, margin = margin(b = 10)),
            plot.subtitle = element_text(size = 11, hjust = 0, color = "#555555", margin = margin(b = 15)),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 10),
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_line(colour = "gray", linewidth = 0.3, linetype = "solid"),
            panel.grid.minor = element_blank(),
            axis.line = element_line(colour = "black", linewidth = 0.5),
            panel.background = element_rect(fill = "white", colour = NA),
            plot.background = element_rect(fill = "white", colour = NA),
            panel.border = element_blank(),
            plot.margin = margin(t = 20, r = 10, b = 10, l = 10),
            legend.position = legend.position,
        )
}

screening_plt <- labels %>%
    rename(plotting_col = my_screening) %>%
    group_by(dataset, plotting_col) %>%
    summarise(
        n = n(),
        .groups = "drop"
    ) %>%
    group_by(dataset) %>%
    mutate(
        prop = n / sum(n) * 100
    ) %>%
    ungroup() %>%
    barplot(
        title="<20% of articles pass screening",
        subtitle="Based on manual review",
        x_label="Screening Outcome",
        legend.position="none"
    )

# %%
priority_plt <- labels %>%
    filter(my_screening == "Pass") %>%
    rename(plotting_col = my_priority) %>%
    group_by(dataset, plotting_col) %>%
    summarise(
        n = n(),
        .groups = "drop"
    ) %>%
    group_by(dataset) %>%
    mutate(
        prop = n / sum(n) * 100
    ) %>%
    ungroup() %>%
        barplot(
        title="Most articles are low priority",
        subtitle="Based on manual review",
        x_label="Priority"
    )

# %%
options(repr.plot.width=10, repr.plot.height=4)
screening_plt + priority_plt + plot_layout(widths = c(1.4, 2))

ggsave("img/r_target_priority.png", width = 10, height = 4)

# %%
scores <- fromJSON("results/0.2_scored_articles.json") %>%
    select(doi, score) %>%
    full_join(labels, by = "doi")

scores_screen_plt <- scores %>%
    ggplot(aes(x = my_screening, y = score, fill = my_screening, alpha = dataset)) +
        geom_boxplot(outlier.shape=NA, width = 0.3) +
        geom_beeswarm(shape=21, size=1) +
        labs(
            title="Scores predict\nscreening outcome",
            subtitle="Based on manual review",
            x="Screening Outcome",
            y="Article Score"
        ) +
        scale_alpha_manual(values = c("Train" = 1.0, "Test" = 0.7)) +
        scale_fill_manual(values = c("#DB444B", "#006BA2"), guide = "none") +
        theme_minimal() +
        theme(
            plot.title = element_text(size = 16, face = "bold", hjust = 0, margin = margin(b = 10)),
            plot.subtitle = element_text(size = 11, hjust = 0, color = "#555555", margin = margin(b = 15)),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 10),
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_line(colour = "gray", linewidth = 0.3, linetype = "solid"),
            panel.grid.minor = element_blank(),
            axis.line = element_line(colour = "black", linewidth = 0.5),
            panel.background = element_rect(fill = "white", colour = NA),
            plot.background = element_rect(fill = "white", colour = NA),
            panel.border = element_blank(),
            plot.margin = margin(t = 20, r = 10, b = 10, l = 10),
            legend.position = "none",
        )

scores_priority_plt <- scores %>%
    filter(my_screening == "Pass") %>%
    ggplot(aes(x = my_priority, y = score, fill = my_priority, alpha=dataset)) +
        geom_boxplot(outlier.shape=NA, width = 0.3) +
        geom_beeswarm(shape=21, size=1) +
        # add lines connecting the means of each priority level
        geom_smooth(
            aes(group=1),
            method="lm",
            formula = y ~ as.numeric(x),
            color="black",
            linetype="dashed",
            se=FALSE,
            size=0.5,
            alpha=0.5
        ) +
        labs(
            title="Scores weakly predict priority",
            subtitle="Based on manual review",
            x="Priority",
            y="Article Score",
            alpha="Dataset"
        ) +
        scale_fill_manual(values = c("#006BA2", "#6c7a89", "#DB444B"), guide = "none") +
        scale_alpha_manual(values = c("Train" = 1.0, "Test" = 0.7)) +
        guides(alpha = guide_legend(override.aes = list(fill = "#DB444B"))) +
        theme_minimal() +
        theme(
            plot.title = element_text(size = 16, face = "bold", hjust = 0, margin = margin(b = 10)),
            plot.subtitle = element_text(size = 11, hjust = 0, color = "#555555", margin = margin(b = 15)),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 10),
            panel.grid.major.x = element_blank(),
            panel.grid.major.y = element_line(colour = "gray", linewidth = 0.3, linetype = "solid"),
            panel.grid.minor = element_blank(),
            axis.line = element_line(colour = "black", linewidth = 0.5),
            panel.background = element_rect(fill = "white", colour = NA),
            plot.background = element_rect(fill = "white", colour = NA),
            panel.border = element_blank(),
            plot.margin = margin(t = 20, r = 10, b = 10, l = 10),
            legend.position = "right",
        )

options(repr.plot.width=8, repr.plot.height=4)
scores_screen_plt + scores_priority_plt + plot_layout(widths = c(1.5, 2))
ggsave("img/r_score_performance.png", width = 8, height = 4)

# %%

# %%

# %%
labels

# %%
