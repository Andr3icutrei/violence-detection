import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import { AbstractControl, FormBuilder, FormGroup, ReactiveFormsModule, ValidationErrors, ValidatorFn, Validators } from '@angular/forms';
import {TranslatePipe} from "@ngx-translate/core";
import { Router } from '@angular/router';
import { CreditsService } from '../../../../../services/credits/credits.service';

@Component({
  selector: 'app-cronjob-credits-form',
  imports: [ReactiveFormsModule, TranslatePipe],
  templateUrl: './cronjob-credits-form.html',
  styleUrl: './cronjob-credits-form.css',
})
export class CronjobCreditsForm implements OnInit {
  creditsCronjobUpdateOriginal!: number;

  form!: FormGroup;

  constructor(
    private formBuilder: FormBuilder,
    private cdr: ChangeDetectorRef,
    private router: Router,
    private creditsService: CreditsService,
  ) {}

  ngOnInit(): void {
    this.loadCreditsInformation();
  }

  private loadCreditsInformation(): void {
    this.creditsService.getCreditsCronjobUpdate().subscribe({
      next: (data: number) => {
        this.creditsCronjobUpdateOriginal = data;
        this.constructForm();
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.router.navigate(['/dashboard']);
      },
    });
  }

  private constructForm(): void {
    this.form = this.formBuilder.group({
      credits: [this.creditsCronjobUpdateOriginal, [Validators.required]],
    }, {
      validators: [this.inputtedValueDiffersFromOriginal()],
    });
  }

  get credits(): number {
    return this.form.get('credits')?.value as number;
  }

  public modifyUpdateCredits(): void {
    this.creditsService.patchCreditsCronjobUpdate(this.credits).subscribe({
      next: (): void => {
        this.loadCreditsInformation();
      }
    })
  }

  private inputtedValueDiffersFromOriginal(): ValidatorFn {
    return (formGroup: AbstractControl): ValidationErrors | null => {
      const creditsControl: number = formGroup.get('credits')?.value as number;
      return creditsControl !== this.creditsCronjobUpdateOriginal ? null : { noChanges: true };
    }
  }
}
