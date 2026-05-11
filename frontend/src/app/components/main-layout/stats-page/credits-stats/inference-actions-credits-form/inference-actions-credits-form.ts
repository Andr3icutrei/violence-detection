import { ChangeDetectorRef, Component, OnInit } from '@angular/core';
import {
  AbstractControl,
  FormArray,
  FormBuilder,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
  ValidationErrors,
  ValidatorFn,
} from '@angular/forms';
import { InferenceActionResponseDto } from '../../../../../core/api/models/inference-action-response-dto';
import { InferenceActionsService } from '../../../../../services/inference_actions/inference-actions.service';
import { TranslatePipe } from '@ngx-translate/core';
import { Router } from '@angular/router';

@Component({
  selector: 'app-inference-actions-credits-form',
  imports: [FormsModule, ReactiveFormsModule, TranslatePipe],
  templateUrl: './inference-actions-credits-form.html',
  styleUrl: './inference-actions-credits-form.css',
})
export class InferenceActionsCreditsForm implements OnInit {
  inferenceActionsOriginal!: InferenceActionResponseDto[];

  form!: FormGroup;

  constructor(
    private cdr: ChangeDetectorRef,
    private formBuilder: FormBuilder,
    private router: Router,
    private inferenceActionsService: InferenceActionsService,
  ) {}

  ngOnInit(): void {
    this.loadInferenceActionsInformation();
  }

  get inferenceActionsArray(): FormArray {
    return this.form.get('inferenceActions') as FormArray;
  }

  public modifyInferenceActionsCredits(): void {
    const actions: { id: number; newCredits: number }[] =
      this.inferenceActionsArray.controls.map((control) => {
        return {
          id: control.get('id')?.value as number,
          newCredits: control.get('credits')?.value as number,
        };
      });
    this.inferenceActionsService.updateCreditsForAction(actions).subscribe({
      next: () => {
        this.loadInferenceActionsInformation();
      },
    });
  }

  private loadInferenceActionsInformation(): void {
    this.inferenceActionsService.getInferenceActionsStats().subscribe({
      next: (data: InferenceActionResponseDto[]) => {
        this.inferenceActionsOriginal = data;
        this.constructForm();
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.router.navigate(['/dashboard']);
      },
    });
  }

  private constructForm() {
    const formGroups = this.inferenceActionsOriginal.map((action) =>
      this.formBuilder.group({
        actionId: [action.action_id],
        credits: [action.credits],
        id: [action.id],
        name: [action.name],
      }),
    );

    this.form = this.formBuilder.group(
      {
        inferenceActions: this.formBuilder.array(formGroups),
      },
      {
        validators: [this.atLeastOneModificationValidators()],
      },
    );
  }

  private atLeastOneModificationValidators(): ValidatorFn {
    return (formGroup: AbstractControl): ValidationErrors | null => {
      const currentActions = formGroup.get('inferenceActions') as FormArray;
      if (!currentActions || !this.inferenceActionsOriginal) {
        return null;
      }
      const hasChanged = currentActions.controls.some((control, index) => {
        const originalCredits = this.inferenceActionsOriginal[index].credits;
        const currentCredits = control.get('credits')?.value;
        return originalCredits !== currentCredits;
      });
      return hasChanged ? null : { noChanges: true };
    };
  }
}
